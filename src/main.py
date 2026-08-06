import asyncio

from viam.module.module import Module

from models.d405_sim import D405Sim as D405SimModel  # noqa: F401

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
