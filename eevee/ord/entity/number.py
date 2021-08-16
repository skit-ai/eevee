from eevee.types import Entity


def eq(a: Entity, b: Entity) -> bool:
    return a["values"] == b["values"]
