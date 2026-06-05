"""Hand-authored room and item templates for dungeon generation."""

from __future__ import annotations

from dataclasses import dataclass

from dungeon_agent.game.domain import Item, ItemRelationType, Room


@dataclass(frozen=True, slots=True)
class RoomTemplate:
    room_id: str
    title: str
    description: str
    search_description: str
    dark_description: str
    has_ambient_light: bool
    room_tags: tuple[str, ...] = ()
    clue_item_ids: tuple[str, ...] = ()

    def to_room(self) -> Room:
        return Room(
            room_id=self.room_id,
            name=self.title,
            description=self.description,
            dark_description=self.dark_description,
            has_ambient_light=self.has_ambient_light,
        )


@dataclass(frozen=True, slots=True)
class RelationPlacementHook:
    anchor_item_id: str
    relation_type: ItemRelationType
    reveal_with_action: str


@dataclass(frozen=True, slots=True)
class ItemTemplate:
    item_id: str
    name: str
    short_description: str
    detail: str
    portable: bool
    searchable: bool
    openable: bool = False
    movable: bool = False
    is_container: bool = False
    starts_open: bool = False
    is_light_source: bool = False
    is_wearable: bool = False
    is_treasure: bool = False
    is_locked: bool = False
    key_item_id: str | None = None
    loot_value: int | None = None
    item_tags: tuple[str, ...] = ()
    preferred_room_tags: tuple[str, ...] = ()
    relation_hooks: tuple[RelationPlacementHook, ...] = ()

    def to_item(self) -> Item:
        return Item(
            item_id=self.item_id,
            name=self.name,
            short_description=self.short_description,
            detail=self.detail,
            portable=self.portable,
            is_container=self.is_container,
            is_open=self.starts_open,
            is_light_source=self.is_light_source,
            is_wearable=self.is_wearable,
            is_locked=self.is_locked,
            key_item_id=self.key_item_id,
            loot_value=self.loot_value if self.loot_value is not None else _infer_loot_value(self),
        )


def _infer_loot_value(template: ItemTemplate) -> int:
    if not template.portable:
        return 0
    if template.is_treasure:
        return 250
    if "key" in template.item_tags:
        return 35
    if "armor" in template.item_tags:
        return 45
    if template.is_light_source:
        return 25
    if "container" in template.item_tags:
        return 20
    return 12


@dataclass(frozen=True, slots=True)
class DiscoveryChainStep:
    step_id: str
    room_id: str
    focus_item_id: str
    action_hint: str
    yields_item_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryChain:
    chain_id: str
    title: str
    objective: str
    steps: tuple[DiscoveryChainStep, ...]


ROOM_TEMPLATES: tuple[RoomTemplate, ...] = (
    RoomTemplate(
        room_id="entry_vestibule",
        title="Entry Vestibule",
        description="Rainwater drips through a cracked dome and streaks the floor with chalky veins.",
        search_description="Boot scuffs cluster near a north arch where someone hesitated before pressing on.",
        dark_description="You trace a shallow chamber by touch while rain ticks somewhere overhead.",
        has_ambient_light=True,
        room_tags=("entry", "transition"),
        clue_item_ids=("chalk_stub", "map_scrap"),
    ),
    RoomTemplate(
        room_id="watchers_gallery",
        title="Watcher’s Gallery",
        description="Blind stone faces stare from alcoves above a hall lined with brittle pennants.",
        search_description="One pennant sags lower than the rest, hiding dustless stone behind it.",
        dark_description="Shapes loom above you, silent and watchful in the dark.",
        has_ambient_light=False,
        room_tags=("observation", "clue"),
        clue_item_ids=("tarnished_tapestry", "vault_key"),
    ),
    RoomTemplate(
        room_id="ashen_cloister",
        title="Ashen Cloister",
        description="A square cloister circles a dead brazier where ash still clings to the mortar.",
        search_description="Charcoal arrows on the floor point toward old storage nooks.",
        dark_description="Dry grit skates under your boots; the air tastes of old smoke.",
        has_ambient_light=False,
        room_tags=("monastic", "storage"),
        clue_item_ids=("flint_striker", "wick_spool"),
    ),
    RoomTemplate(
        room_id="collapsed_refectory",
        title="Collapsed Refectory",
        description="Long tables lie crushed beneath a fallen beam and mounds of slate dust.",
        search_description="A crate corner protrudes from rubble where hands once dug in haste.",
        dark_description="Broken timber hems you in and muffles every step.",
        has_ambient_light=False,
        room_tags=("ruin", "container"),
        clue_item_ids=("mossy_crate", "rune_tablet"),
    ),
    RoomTemplate(
        room_id="smugglers_turn",
        title="Smuggler’s Turn",
        description="The corridor bends sharply around stacked barrels branded with a faded moon sigil.",
        search_description="A hand-width gap behind the barrels carries a cool current from deeper vaults.",
        dark_description="You feel curved walls and narrow footing as the tunnel kinks away.",
        has_ambient_light=False,
        room_tags=("smuggler", "shortcut"),
        clue_item_ids=("crowbar", "rope_coil"),
    ),
    RoomTemplate(
        room_id="well_of_whispers",
        title="Well of Whispers",
        description="A dry well yawns at center, its throat lined with carved prayers and old claw marks.",
        search_description="A frayed rope still bites into the pulley wheel above the shaft.",
        dark_description="Breath echoes in a deep shaft you cannot see.",
        has_ambient_light=False,
        room_tags=("well", "echo"),
        clue_item_ids=("well_bucket", "chain_hook"),
    ),
    RoomTemplate(
        room_id="lantern_workshop",
        title="Lantern Workshop",
        description="Benches of dented brass parts surround a soot-black forge that has gone cold.",
        search_description="Fresh oil scent lingers around one sealed lamp kept apart from the scrap.",
        dark_description="Metal edges scrape your sleeves in a room full of cramped benches.",
        has_ambient_light=True,
        room_tags=("workshop", "light"),
        clue_item_ids=("ember_lantern", "flint_striker"),
    ),
    RoomTemplate(
        room_id="scribe_archive",
        title="Scribe Archive",
        description="Shelves of wax-stiff scrolls stand shoulder to shoulder beneath a painted night sky.",
        search_description="A ledger margin repeats the phrase 'keys bow to mirrored vows.'",
        dark_description="Paper rustles softly, but the shelves are only silhouettes.",
        has_ambient_light=True,
        room_tags=("archive", "lore"),
        clue_item_ids=("scribe_quill", "obsidian_mirror"),
    ),
    RoomTemplate(
        room_id="flooded_crypt",
        title="Flooded Crypt",
        description="Shallow black water ripples between stone biers etched with drowned family names.",
        search_description="One sarcophagus lid sits crooked, as if something pried it open before.",
        dark_description="Cold water laps your ankles while stone coffins loom unseen.",
        has_ambient_light=False,
        room_tags=("crypt", "water"),
        clue_item_ids=("stone_sarcophagus", "silver_chalice"),
    ),
    RoomTemplate(
        room_id="iron_gatehouse",
        title="Iron Gatehouse",
        description="Rust-rimmed portcullis teeth hang overhead, frozen halfway through an old retreat.",
        search_description="A prayer chest is wedged beneath the winch where guards once knelt.",
        dark_description="Metal groans faintly above, heavy and unresolved.",
        has_ambient_light=True,
        room_tags=("gate", "lock"),
        clue_item_ids=("prayer_chest", "coffer_key"),
    ),
    RoomTemplate(
        room_id="moonwell_sanctum",
        title="Moonwell Sanctum",
        description="A shallow basin reflects pale light from a slit in the ceiling like liquid silver.",
        search_description="Offerings around the rim include a chalice and a key-shaped wax imprint.",
        dark_description="The chamber feels open, but only a dim shimmer marks its center.",
        has_ambient_light=True,
        room_tags=("sanctum", "ritual"),
        clue_item_ids=("silver_chalice", "wax_seal"),
    ),
    RoomTemplate(
        room_id="ossuary_walk",
        title="Ossuary Walk",
        description="Rows of skull niches climb the walls, each socket packed with black candle stubs.",
        search_description="One niche is empty except for drag marks leading behind a cracked statue.",
        dark_description="Dry bone edges crowd the passage and swallow your voice.",
        has_ambient_light=False,
        room_tags=("ossuary", "statue"),
        clue_item_ids=("weathered_statue", "vault_key"),
    ),
    RoomTemplate(
        room_id="forgotten_armory",
        title="Forgotten Armory",
        description="Weapon racks rot in place while a watchman’s helm keeps mute vigil on a pike.",
        search_description="A hidden slot behind a rack perfectly fits a narrow iron key.",
        dark_description="Cold iron surrounds you, but details vanish beyond arm’s reach.",
        has_ambient_light=False,
        room_tags=("armory", "martial"),
        clue_item_ids=("watchman_helm", "iron_spike"),
    ),
    RoomTemplate(
        room_id="vault_antechamber",
        title="Vault Antechamber",
        description="A circular room of polished basalt channels every sound toward a sealed bronze door.",
        search_description="The lockplate bears twin symbols matching two different key wards.",
        dark_description="Your footsteps return sharply from smooth walls around a hidden door.",
        has_ambient_light=False,
        room_tags=("vault", "lock"),
        clue_item_ids=("ironbound_coffer", "coffer_key"),
    ),
    RoomTemplate(
        room_id="sunken_treasure_vault",
        title="Sunken Treasure Vault",
        description="Broken pedestals ring a sunken dais where gold dust glitters in stale air.",
        search_description="An ironbound coffer sits at the dais center, chained but untouched.",
        dark_description="The room falls away beneath you and swallows every glint.",
        has_ambient_light=False,
        room_tags=("treasure", "goal"),
        clue_item_ids=("ironbound_coffer", "sunshard_treasure"),
    ),
)


ITEM_TEMPLATES: tuple[ItemTemplate, ...] = (
    ItemTemplate(
        item_id="ember_lantern",
        name="Ember Lantern",
        short_description="a brass lantern hangs above the bench",
        detail="A brass lantern with a protected wick and a thumb latch for quick lighting.",
        portable=True,
        searchable=False,
        is_light_source=True,
        item_tags=("light_source", "quest_critical"),
        preferred_room_tags=("workshop", "light"),
    ),
    ItemTemplate(
        item_id="flint_striker",
        name="Flint Striker",
        short_description="a flint striker rests in an ash tray",
        detail="Two steel prongs and a worn flint nub produce a sharp spray of sparks.",
        portable=True,
        searchable=False,
        item_tags=("tool", "fire"),
        preferred_room_tags=("workshop", "monastic"),
    ),
    ItemTemplate(
        item_id="wick_spool",
        name="Wick Spool",
        short_description="a spool of lamp wick sits under a rag",
        detail="Oiled thread wrapped around cedar, enough to service several lamps.",
        portable=True,
        searchable=True,
        item_tags=("supply",),
        preferred_room_tags=("workshop", "storage"),
    ),
    ItemTemplate(
        item_id="chalk_stub",
        name="Chalk Stub",
        short_description="a snapped chalk stub lies near the doorway",
        detail="Soft white chalk used for trail marks and hurried arrows.",
        portable=True,
        searchable=False,
        item_tags=("clue",),
        preferred_room_tags=("entry", "transition"),
    ),
    ItemTemplate(
        item_id="rope_coil",
        name="Rope Coil",
        short_description="a damp rope coil hangs from an iron peg",
        detail="A twenty-foot hemp rope with frayed but serviceable ends.",
        portable=True,
        searchable=False,
        item_tags=("tool",),
        preferred_room_tags=("well", "smuggler"),
    ),
    ItemTemplate(
        item_id="crowbar",
        name="Crowbar",
        short_description="a bent crowbar leans against the barrels",
        detail="Heavy leverage tool, perfect for prying boards or stone lids.",
        portable=True,
        searchable=False,
        item_tags=("tool", "force"),
        preferred_room_tags=("smuggler", "ruin"),
    ),
    ItemTemplate(
        item_id="bone_needle",
        name="Bone Needle",
        short_description="a carved bone needle is tucked in a seam",
        detail="A slender bone needle engraved with tiny warding marks.",
        portable=True,
        searchable=True,
        item_tags=("trinket",),
        preferred_room_tags=("archive", "ossuary"),
    ),
    ItemTemplate(
        item_id="copper_token",
        name="Copper Token",
        short_description="a copper token glints in pooled water",
        detail="A stamped token bearing a moon crest and a gate number.",
        portable=True,
        searchable=True,
        item_tags=("currency", "clue"),
        preferred_room_tags=("well", "crypt"),
    ),
    ItemTemplate(
        item_id="rune_tablet",
        name="Rune Tablet",
        short_description="a clay rune tablet peeks from a crate",
        detail="Cracked clay tablet warning that treasure sleeps behind double locks.",
        portable=True,
        searchable=True,
        item_tags=("clue", "lore"),
        preferred_room_tags=("archive", "ruin"),
        relation_hooks=(
            RelationPlacementHook(
                anchor_item_id="mossy_crate",
                relation_type=ItemRelationType.IN,
                reveal_with_action="open",
            ),
        ),
    ),
    ItemTemplate(
        item_id="iron_spike",
        name="Iron Spike",
        short_description="an iron spike rests in a weapon rack",
        detail="A tempered spike once used to brace gates and hold doors.",
        portable=True,
        searchable=False,
        item_tags=("tool",),
        preferred_room_tags=("armory", "gate"),
    ),
    ItemTemplate(
        item_id="well_bucket",
        name="Well Bucket",
        short_description="a dented bucket hangs over the dry shaft",
        detail="A bucket patched with rivets and marked by rope burns.",
        portable=False,
        searchable=True,
        movable=True,
        item_tags=("fixture",),
        preferred_room_tags=("well",),
    ),
    ItemTemplate(
        item_id="salted_ration",
        name="Salted Ration",
        short_description="a wrapped ration packet sits on a ledge",
        detail="Dried meat and hard biscuit sealed in waxed cloth.",
        portable=True,
        searchable=False,
        item_tags=("supply",),
        preferred_room_tags=("smuggler", "entry"),
    ),
    ItemTemplate(
        item_id="map_scrap",
        name="Map Scrap",
        short_description="a torn map scrap clings beneath cloth",
        detail="A partial map naming the sanctum and a marked path to the vault antechamber.",
        portable=True,
        searchable=True,
        item_tags=("clue", "quest_critical"),
        preferred_room_tags=("entry", "observation"),
        relation_hooks=(
            RelationPlacementHook(
                anchor_item_id="tarnished_tapestry",
                relation_type=ItemRelationType.UNDER,
                reveal_with_action="move",
            ),
        ),
    ),
    ItemTemplate(
        item_id="obsidian_mirror",
        name="Obsidian Mirror",
        short_description="an obsidian mirror rests in a cedar case",
        detail="A polished black mirror that reflects candlelight as pale silver rings.",
        portable=True,
        searchable=False,
        item_tags=("ritual", "clue"),
        preferred_room_tags=("archive", "sanctum"),
    ),
    ItemTemplate(
        item_id="watchman_helm",
        name="Watchman Helm",
        short_description="a watchman’s helm balances on a rusted pike",
        detail="Iron helmet with narrowed eye slits and dried leather lining.",
        portable=True,
        searchable=False,
        is_wearable=True,
        item_tags=("armor",),
        preferred_room_tags=("armory", "gate"),
    ),
    ItemTemplate(
        item_id="scribe_quill",
        name="Scribe Quill",
        short_description="a black quill lies beside an inkstone",
        detail="Crow-feather quill stiffened with lacquer and silver wire.",
        portable=True,
        searchable=False,
        item_tags=("lore",),
        preferred_room_tags=("archive",),
    ),
    ItemTemplate(
        item_id="wax_seal",
        name="Wax Seal",
        short_description="a moon-stamped wax seal is pressed into a tray",
        detail="Seal impression matching symbols on the vault lockplate.",
        portable=True,
        searchable=True,
        item_tags=("clue", "ritual"),
        preferred_room_tags=("archive", "sanctum"),
    ),
    ItemTemplate(
        item_id="silver_chalice",
        name="Silver Chalice",
        short_description="a silver chalice catches the faint light",
        detail="A ritual chalice tarnished by mineral-rich crypt water.",
        portable=True,
        searchable=True,
        item_tags=("ritual",),
        preferred_room_tags=("crypt", "sanctum"),
    ),
    ItemTemplate(
        item_id="mason_hammer",
        name="Mason Hammer",
        short_description="a mason’s hammer lies beside loose stone",
        detail="Compact stone hammer with one flat face and one pointed chisel edge.",
        portable=True,
        searchable=False,
        item_tags=("tool",),
        preferred_room_tags=("ruin", "entry"),
    ),
    ItemTemplate(
        item_id="chain_hook",
        name="Chain Hook",
        short_description="a chain hook dangles from the pulley axle",
        detail="Hooked iron segment for dragging heavy loads across stone.",
        portable=True,
        searchable=True,
        item_tags=("tool", "force"),
        preferred_room_tags=("well", "gate"),
    ),
    ItemTemplate(
        item_id="weathered_statue",
        name="Weathered Statue",
        short_description="a weathered statue blocks a shallow recess",
        detail="A cracked guardian statue on a grinding plinth, heavy but shiftable.",
        portable=False,
        searchable=True,
        movable=True,
        item_tags=("anchor", "obstacle"),
        preferred_room_tags=("ossuary", "observation"),
    ),
    ItemTemplate(
        item_id="mossy_crate",
        name="Mossy Crate",
        short_description="a mossy crate is wedged beneath fallen timber",
        detail="Water-swollen crate bound with bronze straps and a stiff lid.",
        portable=False,
        searchable=True,
        openable=True,
        is_container=True,
        item_tags=("anchor", "container"),
        preferred_room_tags=("ruin", "storage"),
    ),
    ItemTemplate(
        item_id="prayer_chest",
        name="Prayer Chest",
        short_description="a prayer chest rests beneath the winch",
        detail="A cedar chest with iron corners and a narrow lockplate.",
        portable=False,
        searchable=True,
        openable=True,
        is_container=True,
        is_locked=True,
        key_item_id="vault_key",
        item_tags=("container", "lock"),
        preferred_room_tags=("gate", "sanctum"),
    ),
    ItemTemplate(
        item_id="ironbound_coffer",
        name="Ironbound Coffer",
        short_description="an ironbound coffer sits on the dais",
        detail="A thick-lidded coffer with twin key wards and a recessed clasp.",
        portable=False,
        searchable=True,
        openable=True,
        is_container=True,
        is_locked=True,
        key_item_id="coffer_key",
        item_tags=("container", "lock", "quest_critical"),
        preferred_room_tags=("vault", "treasure"),
    ),
    ItemTemplate(
        item_id="vault_key",
        name="Vault Key",
        short_description="a narrow iron key glints in dust",
        detail="A long-shafted key etched with the mark of the outer vault ward.",
        portable=True,
        searchable=True,
        item_tags=("key", "quest_critical"),
        preferred_room_tags=("ossuary", "observation"),
        relation_hooks=(
            RelationPlacementHook(
                anchor_item_id="weathered_statue",
                relation_type=ItemRelationType.BEHIND,
                reveal_with_action="move",
            ),
        ),
    ),
    ItemTemplate(
        item_id="coffer_key",
        name="Coffer Key",
        short_description="a broad brass key lies in a felt-lined slot",
        detail="A brass key that fits the inner ward of the treasure coffer.",
        portable=True,
        searchable=True,
        item_tags=("key", "quest_critical"),
        preferred_room_tags=("gate", "vault"),
        relation_hooks=(
            RelationPlacementHook(
                anchor_item_id="prayer_chest",
                relation_type=ItemRelationType.IN,
                reveal_with_action="open",
            ),
        ),
    ),
    ItemTemplate(
        item_id="sunshard_treasure",
        name="Sunshard",
        short_description="a faceted sunshard pulses with trapped dawnlight",
        detail="A crystal relic warm to the touch, bright enough to cast sharp shadows.",
        portable=True,
        searchable=True,
        is_treasure=True,
        item_tags=("treasure", "quest_critical"),
        preferred_room_tags=("treasure", "vault"),
        relation_hooks=(
            RelationPlacementHook(
                anchor_item_id="ironbound_coffer",
                relation_type=ItemRelationType.IN,
                reveal_with_action="open",
            ),
        ),
    ),
    ItemTemplate(
        item_id="amber_reliquary",
        name="Amber Reliquary",
        short_description="an amber reliquary hangs from a chain",
        detail="A translucent reliquary holding brittle petals and ash.",
        portable=True,
        searchable=True,
        openable=True,
        is_container=True,
        item_tags=("container", "ritual"),
        preferred_room_tags=("sanctum", "archive"),
    ),
    ItemTemplate(
        item_id="stone_sarcophagus",
        name="Stone Sarcophagus",
        short_description="a stone sarcophagus rests with its lid ajar",
        detail="A carved sarcophagus with a heavy lid that can be shifted further.",
        portable=False,
        searchable=True,
        openable=True,
        movable=True,
        is_container=True,
        starts_open=True,
        item_tags=("anchor", "container"),
        preferred_room_tags=("crypt",),
    ),
    ItemTemplate(
        item_id="tarnished_tapestry",
        name="Tarnished Tapestry",
        short_description="a tarnished tapestry hangs crooked on iron hooks",
        detail="Heavy woven cloth hiding scratches on the wall behind it.",
        portable=False,
        searchable=True,
        movable=True,
        item_tags=("anchor", "obstacle"),
        preferred_room_tags=("observation", "entry"),
    ),
)


DISCOVERY_CHAINS: tuple[DiscoveryChain, ...] = (
    DiscoveryChain(
        chain_id="chain-light",
        title="Spark Before Shadows",
        objective="Secure a reliable light source before deep exploration.",
        steps=(
            DiscoveryChainStep(
                step_id="light-clue-archive",
                room_id="scribe_archive",
                focus_item_id="wax_seal",
                action_hint="Search the shelves for notes that mention the workshop lamp.",
                yields_item_id="ember_lantern",
            ),
            DiscoveryChainStep(
                step_id="light-recover-lantern",
                room_id="lantern_workshop",
                focus_item_id="ember_lantern",
                action_hint="Take the lantern and prime it with striker and wick.",
            ),
        ),
    ),
    DiscoveryChain(
        chain_id="chain-treasure",
        title="Twin Wards of the Vault",
        objective="Retrieve the Sunshard by resolving layered relation and lock gates.",
        steps=(
            DiscoveryChainStep(
                step_id="treasure-move-statue",
                room_id="ossuary_walk",
                focus_item_id="weathered_statue",
                action_hint="Move the statue to reveal the key hidden behind it.",
                yields_item_id="vault_key",
            ),
            DiscoveryChainStep(
                step_id="treasure-open-prayer-chest",
                room_id="iron_gatehouse",
                focus_item_id="prayer_chest",
                action_hint="Use the vault key to unlock and open the prayer chest.",
                yields_item_id="coffer_key",
            ),
            DiscoveryChainStep(
                step_id="treasure-open-coffer",
                room_id="sunken_treasure_vault",
                focus_item_id="ironbound_coffer",
                action_hint="Use the coffer key on the ironbound coffer and take the Sunshard.",
                yields_item_id="sunshard_treasure",
            ),
        ),
    ),
)
