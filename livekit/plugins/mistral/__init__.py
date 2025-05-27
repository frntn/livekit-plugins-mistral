# Copyright 2025 Matthieu FRONTON
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mistral AI plugin for LiveKit Agents
Supports Large Language Models (LLMs) with Mistral AI.
"""

from .llm import LLM, LLMStream, AgentError
from .log import logger
from .models import ChatModels 
from .version import __version__

__all__ = [
    "LLM",
    "LLMStream",
    "AgentError",
    "ChatModels",
    "logger",
    "__version__",
]

from livekit.agents import Plugin

class MistralPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        # Mistral AI is API-based, so typically no local files need downloading.
        # This method can be left as pass or used if specific resources were needed.
        pass

Plugin.register_plugin(MistralPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]
__pdoc__ = {}
for n in NOT_IN_ALL:
    __pdoc__[n] = False
