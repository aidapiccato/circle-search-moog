"""Task with predators chasing agent in open arena.
The predators (red circles) chase the agent. The predators bouce off the arena
boundaries, while the agent cannot exit but does not bounce (i.e. it has
inelastic collisions with the boundaries). Trials only terminate when the agent
is caught by a predator. The subject controls the agent with a joystick.
This task also contains an auto-curriculum: When the subject does well (evades
the predators for a long time before being caught), the predators' masses are
decreased, thereby increasing the predators' speeds. Conversely, when the
subject does poorly (gets caught quickly), the predators' masses are increased,
thereby decreasing the predators' speeds.
"""

import collections
import numpy as np
import os
import time

from moog import action_spaces
from moog import game_rules
from moog import observers
from moog import physics as physics_lib
from moog import shapes
from moog import tasks
from moog import sprite

from rules import ModifyWhenContacting
from rules import ModifyStateOnContact

_RED = [0, 1.0, 1.0]
_GREEN = [0.3, 1.0, 1.0]
_BLUE = [0.7, 1.0, 1.0]
_YELLOW = [0.18, .98, 1.0]
_COLORS = [_RED, _GREEN, _BLUE, _YELLOW]
_GRAY = [0, 0, 0.7]
_MAX_CONTACTS = 3
_N_CIRCLES = 18


def _make_opaque(s):
    s.opacity = 255

def _make_transparent(s):
    s.opacity = 0

def _set_color(s, color):
    s.c0 = color[0]
    s.c1 = color[1]
    s.c2 = color[2]


def _color_fruit(s):
    _set_color(s, s.metadata['color'])


def _hide_fruit(s):
    _set_color(s, _GRAY)


class StateInitialization():
    """State initialization class to dynamically adapt predator mass.

    This is essentially an auto-curriculum: When the subject does well (evades
    the predators for a long time before being caught), the predators' masses
    are decreased, thereby increasing the predators' speeds. Conversely, when
    the subject does poorly (gets caught quickly), the predators' masses are
    increased, thereby decreasing the predators' speeds.
    """

    def __init__(self):
        """Constructor.
        This class uses the meta-state to keep track of the number of steps
        before the agent is caught. See the game rules section near the bottom
        of this file for the counter incrementer.
        Args:
            step_scaling_factor: Float. Fractional decrease of predator mass
                after a trial longer than threshold_trial_len. Also used as
                fractional increase of predator mass after a trial shorter than
                threshold_trial_len. Should be small and positive.
            threshold_trial_len: Length of a trial above which the predator
                mass is decreased and below which the predator mass is
                increased.
        """

        # Agent

        def _agent_generator():
            return sprite.Sprite(x=0.5, y=0.5, shape='circle', scale=0.1, c0=0.33, c1=1., c2=0.66)

        self._agent_generator = _agent_generator

        # Fruits
        def _map_fruit_generator(color_regions, colors):
            fruit_factors = {'shape': 'circle', 'scale': 0.05}
            fruits_rad = np.linspace(0 + (2 * np.pi / _N_CIRCLES), 2 * np.pi, _N_CIRCLES)
            fruits_grid_x = 0.2 * np.cos(fruits_rad) + 0.5
            fruits_grid_y = 0.2 * np.sin(fruits_rad) + 0.5

            fruits_colors = np.hstack([np.repeat(color, color_regions[color]) for color in np.arange(
                len(colors))])
            fruits_colors = [colors[color] for color in fruits_colors]
            fruits_props = zip(fruits_grid_x, fruits_grid_y, fruits_colors)
            fruit_sprites = [
                sprite.Sprite(x=x, y=y, c0=c0, c1=c1, c2=c2, **fruit_factors)
                for (x, y, (c0, c1, c2)) in fruits_props
            ]

            return fruit_sprites

        def _fruit_generator(target_color, color_regions, colors):
            fruit_factors = {'shape': 'circle', 'scale': 0.05}
            fruits_rad = np.linspace(0 + (2 * np.pi / _N_CIRCLES), 2 * np.pi, _N_CIRCLES)
            fruits_grid_x = 0.35 * np.cos(fruits_rad) + 0.5
            fruits_grid_y = 0.35 * np.sin(fruits_rad) + 0.5
            # Creating transform (currently just rotation)
            rot = np.random.randint(low=0, high=len(fruits_grid_x) - 1, size=1)
            fruits_grid_x = np.roll(fruits_grid_x, rot)
            fruits_grid_y = np.roll(fruits_grid_y, rot)
            fruits_colors = np.hstack([np.repeat(color, color_regions[color]) for color in np.arange(
                len(colors))])
            fruits_colors = [colors[color] for color in fruits_colors]
            fruits_props = zip(fruits_grid_x, fruits_grid_y, fruits_colors)
            fruit_sprites = [
                sprite.Sprite(x=x, y=y, c0=0, c1=0, c2=0.7,
                              metadata={'color': (c0, c1, c2), 'target': (list((c0, c1, c2)) == target_color)},
                              **fruit_factors)
                for (x, y, (c0, c1, c2)) in fruits_props
            ]

            return fruit_sprites


        def _cue_generator(target_color):
            cue = sprite.Sprite(x=0.5, y=0.5, shape='circle', scale=0.1, opacity=0)
            _set_color(cue, target_color)
            return cue


        self._cue_generator = _cue_generator
        self._fruit_generator = _fruit_generator
        self._map_fruit_generator = _map_fruit_generator


        self._walls = shapes.border_walls(
            visible_thickness=0., c0=0., c1=0., c2=0.5)

        self._meta_state = None

        self._agent = None

        self._target = None

    def state_initializer(self):
        """State initializer method to be fed to environment."""

        # Generating agent
        self._agent = self._agent_generator()

        self._n_colors = np.random.randint(low=2, high=len(_COLORS), size=1)
        self._colors = np.random.choice(len(_COLORS), size=self._n_colors).astype(int)
        self._colors = [_COLORS[color] for color in self._colors]

        self._target = self._colors[np.random.choice(len(self._colors), size=1).astype(int)[0]]
        # Generating color regions
        self._color_regions = np.random.multinomial(_N_CIRCLES, np.ones(self._n_colors) / self._n_colors, size=1)[0]

        # Generating map fruits
        self._map_fruits = self._map_fruit_generator(self._color_regions, self._colors)

        # Generating fruits
        self._fruits = self._fruit_generator(self._target, self._color_regions, self._colors)

        # Generating cue
        self._cue = self._cue_generator(self._target)
        state = collections.OrderedDict([
            # ('target', self._target),
            ('walls', self._walls),
            ('contacted_fruits', set()),
            ('cue', (self._cue, )),
            ('agent', (self._agent, )),
            ('fruits', self._fruits),
            ('map_fruits', self._map_fruits)
        ])

        return state

    def meta_state_initializer(self):
        """Meta-state initializer method to be fed to environment."""
        self._meta_state = {'phase': ''}
        return self._meta_state


def get_config(level):
    """Get config dictionary of kwargs for environment constructor.

    Args:
        level: Int. Number of circles.
    """

    ############################################################################
    # Sprite initialization
    ############################################################################

    state_initialization = StateInitialization(
        # TODO: Add structure transformation randomness
    )

    ############################################################################
    # Physics
    ############################################################################

    agent_friction_force = physics_lib.Drag(coeff_friction=0.25)

    forces = (
        (agent_friction_force, 'agent'),
    )

    physics = physics_lib.Physics(*forces, updates_per_env_step=10)

    ############################################################################
    # Task
    ############################################################################

    def _eq_sprite_color(s, color):
        return list((s.c0, s.c1, s.c2)) == list(color)

    def _fruit_reward_fn(_, fruit_sprite):
        hit_target = fruit_sprite.metadata['target']
        return 1 * hit_target + -1 * (not hit_target)

    contact_task = tasks.ContactReward(
        reward_fn=_fruit_reward_fn, layers_0='agent', layers_1='fruits')

    def _should_reset(state, meta_state):
        for s in state['fruits']:
            if s.overlaps_sprite(state['agent'][0]) and s.metadata['target']:
                return True
        if len(state['contacted_fruits']) >= _MAX_CONTACTS:
            return True
        return False

    reset_task = tasks.Reset(condition=_should_reset, steps_after_condition=15)

    task = tasks.CompositeTask(contact_task, reset_task)
    ############################################################################
    # Action space
    ############################################################################

    # action_space = action_spaces.Joystick(
    #     scaling_factor=0.01, action_layers='agent')

    action_space = action_spaces.Grid(action_layers='agent', scaling_factor=0.01)
    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(64, 64), anti_aliasing=1, color_to_rgb='hsv_to_rgb')

    ############################################################################
    # Game rules
    ############################################################################

    def _update_contacted_fruits(state, contacted_fruits):
        state['contacted_fruits'] = state['contacted_fruits'].union(set(contacted_fruits))

    update_rule = ModifyStateOnContact(layers_0='agent', layers_1='fruits', modifier=_update_contacted_fruits)

    contact_rule = ModifyWhenContacting(
        'fruits', 'agent', on_contact=_color_fruit, off_contact=_hide_fruit)

    def _re_center(agent):
        agent.position = (0.5, 0.5)

    center_rule = game_rules.ModifyOnContact(
        'agent', 'fruits', modifier_0=_re_center
    )
    # What should phase sequence look like
    # screen_phase (target cue, all else hidden) -> search phase (all visible)
    # during the search phase -> on contact there is some delay, then the
    show_cue = game_rules.ModifySprites('cue', _make_opaque)
    hide_fruits = game_rules.ModifySprites('fruits', _make_transparent)
    hide_agent = game_rules.ModifySprites('agent', _make_transparent)
    hide_map_fruits = game_rules.ModifySprites('map_fruits', _make_transparent)
    screen_phase = game_rules.Phase(one_time_rules=(show_cue, hide_fruits, hide_agent, hide_map_fruits), duration=40,
                                    name='screen')

    # Search phase
    hide_cue = game_rules.ModifySprites('cue', _make_transparent)
    show_fruits = game_rules.ModifySprites('fruits', _make_opaque)
    show_agent = game_rules.ModifySprites('agent', _make_opaque)
    show_map_fruits = game_rules.ModifySprites('map_fruits', _make_opaque)
    continual_rules = (contact_rule, update_rule, )
    search_phase = game_rules.Phase(one_time_rules=(hide_cue, show_fruits, show_agent, show_map_fruits),
                                    continual_rules=continual_rules,
                                    name='search')

    phase_sequence = game_rules.PhaseSequence(screen_phase, search_phase)
    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initialization.state_initializer,
        'physics': physics,
        'game_rules': (phase_sequence,),
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer},
        'meta_state_initializer': state_initialization.meta_state_initializer,
    }
    return config
