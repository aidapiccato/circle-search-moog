from moog import game_rules
import itertools

class ModifyStateOnContact(game_rules.abstract_rule.AbstractRule):
    """Modify state if sprites contact each other."""

    def __init__(self,
                 layers_0,
                 layers_1,
                 modifier=None,
                 filter_0=None,
                 filter_1=None,):
        """Constructor.
        Applies modifier to state if contact between sprites in layers_0 and layers_1 and those sprites satisfy
        respective filters.

        Args:
            layers_0: String or iterable of strings. Must be layer name(s) in
                environment state.
            layers_1: String or iterable of strings. Must be layer name(s) in
                environment state.
            modifier: Function taking in state and list of contacts in second layer sand modifying in place.
            filter_0: Function taking in a sprite and returning bool.
            filter_1: Function taking in a sprite and returning bool.
        """
        if not isinstance(layers_0, (list, tuple)):
            layers_0 = (layers_0,)
        self._layers_0 = layers_0
        if not isinstance(layers_1, (list, tuple)):
            layers_1 = (layers_1,)
        self._layers_1 = layers_1

        self._modifier = modifier
        if filter_0 is None:
            filter_0 = lambda x: True
        self._filter_0 = filter_0
        if filter_1 is None:
            filter_1 = lambda x: True
        self._filter_1 = filter_1


    def _modify_symmetric(self, state, sprites_0, sprites_1, modifier, filter_0, filter_1):
        if modifier is not None:
            filtered_sprites_0 = [s for s in sprites_0 if filter_0(s)]
            filtered_sprites_1 = [s for s in sprites_1 if filter_1(s)]
            for s_0 in filtered_sprites_0:
                contacts = [s_1 for s_1 in filtered_sprites_1 if s_0.overlaps_sprite(s_1) and id(s_1) != id(s_0)]
                if any(contacts):
                    modifier(state, contacts)

    def step(self, state, meta_state):
        """Apply rule to state."""
        del meta_state

        sprites_0 = list(itertools.chain(*[state[k] for k in self._layers_0]))
        sprites_1 = list(itertools.chain(*[state[k] for k in self._layers_1]))

        self._modify_symmetric(state, sprites_0, sprites_1, self._modifier, self._filter_0, self._filter_1)

class ModifyWhenContacting(game_rules.AbstractRule):
    def __init__(self,
                 modifying_layer,
                 contacting_layer,
                 on_contact,
                 off_contact):
        """Constructor.
        Args:
            modifying_layer: String. Key in environment state. All sprites in
                this layer will be modified when contacting any sprite in
                contacting_layer.
            contacting_layer: String. Key in environment state.
            on_contact: In-place function of sprite. Applied to sprites in
                modifying_layer when they contact a sprite in contacting_layer.
            off_contact: In-place function of sprite. Applied to sprites in
                modifying_layer when they stop contacting sprites in
                contacting_layer.
        """
        self._modifying_layer = modifying_layer
        self._contacting_layer = contacting_layer
        self._on_contact = on_contact
        self._off_contact = off_contact

    def reset(self, state, meta_state):
        del state
        del meta_state
        # This is just to save a tiny bit of efficiency, by remembering which
        # sprites are contacting so we don't apply on_contact and off_contact
        # when we don't need to.
        self._contacting_sprite_ids = set()

    def step(self, state, meta_state):
        """Apply rule to state."""
        del meta_state
        for sprite in state[self._modifying_layer]:
            contacting = any([
                sprite.overlaps_sprite(s) for s in state[self._contacting_layer]
            ])
            if sprite.id in self._contacting_sprite_ids and not contacting:
                self._off_contact(sprite)
                self._contacting_sprite_ids.remove(sprite.id)
            if sprite.id not in self._contacting_sprite_ids and contacting:
                self._on_contact(sprite)
                self._contacting_sprite_ids.add(sprite.id)