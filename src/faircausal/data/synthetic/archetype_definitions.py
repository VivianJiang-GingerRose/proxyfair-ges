from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ArchetypeSpec:
    archetype_id: int
    name: str
    description: str
    all_variables: Tuple[str, ...]
    observed_variables: Tuple[str, ...]
    edges: Tuple[Tuple[str, str], ...]
    latent_variables: Tuple[str, ...]
    protected_variables: Tuple[str, ...]
    outcome_variable: str
    proxy_variable: str
    legit_variable: str
    proxy_variables: Tuple[str, ...] = ()
    legit_variables: Tuple[str, ...] = ()


ARCHETYPES: Dict[int, ArchetypeSpec] = {
    1: ArchetypeSpec(
        archetype_id=1,
        name="selective_suppression",
        description=(
            "S -> V -> Y, Z -> Y. "
            "V is a proxy (discriminatory path); Z is a legitimate predictor independent of S."
        ),
        all_variables=("S", "V", "Z", "Y"),
        observed_variables=("S", "V", "Z", "Y"),
        edges=(("S", "V"), ("V", "Y"), ("Z", "Y")),
        latent_variables=(),
        protected_variables=("S",),
        outcome_variable="Y",
        proxy_variable="V",
        legit_variable="Z",
        proxy_variables=("V",),
        legit_variables=("Z",),
    ),
    3: ArchetypeSpec(
        archetype_id=3,
        name="confounder",
        description=(
            "U -> S, U -> V, V -> Y, Z -> Y with latent U. "
            "V's association with S is confounded (not causal); Z is a legitimate predictor independent of S."
        ),
        all_variables=("U", "S", "V", "Z", "Y"),
        observed_variables=("S", "V", "Z", "Y"),
        edges=(("U", "S"), ("U", "V"), ("V", "Y"), ("Z", "Y")),
        latent_variables=("U",),
        protected_variables=("S",),
        outcome_variable="Y",
        proxy_variable="V",
        legit_variable="Z",
        proxy_variables=("V",),
        legit_variables=("Z",),
    ),
    4: ArchetypeSpec(
        archetype_id=4,
        name="multi_attribute_displacement",
        description=(
            "C -> S1, C -> S2; S1 -> V1 -> Y and S2 -> V2 -> Y; Z -> Y. "
            "C induces correlation between protected attributes, V1 and V2 are "
            "attribute-specific proxy paths, and Z is a legitimate predictor."
        ),
        all_variables=("C", "S1", "S2", "V1", "V2", "Z", "Y"),
        observed_variables=("C", "S1", "S2", "V1", "V2", "Z", "Y"),
        edges=(
            ("C", "S1"),
            ("C", "S2"),
            ("S1", "V1"),
            ("S2", "V2"),
            ("V1", "Y"),
            ("V2", "Y"),
            ("Z", "Y"),
        ),
        latent_variables=(),
        protected_variables=("S1", "S2"),
        outcome_variable="Y",
        proxy_variable="V1",
        legit_variable="Z",
        proxy_variables=("V1", "V2"),
        legit_variables=("Z",),
    ),
}


def get_archetype(archetype_id: int) -> ArchetypeSpec:
    if archetype_id not in ARCHETYPES:
        raise ValueError(f"Unknown archetype_id={archetype_id}. Expected one of {sorted(ARCHETYPES)}")
    return ARCHETYPES[archetype_id]


def observed_mask(archetype: ArchetypeSpec) -> List[bool]:
    observed = set(archetype.observed_variables)
    return [var in observed for var in archetype.all_variables]
