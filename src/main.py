import asyncio

from viam.module.module import Module

# Importing a model registers it. Both models this module offers are listed
# here, and both are served by the same process.
from models.d405 import D405 as D405Model                      # noqa: F401
from models.d405_sim import D405Sim as D405SimModel            # noqa: F401

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
