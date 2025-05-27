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

from typing import Literal

# for model description https://mistral.ai/models
# for model id references and version https://docs.mistral.ai/getting-started/models/models_overview/#api-versioning
ChatModels = Literal[
    "mistral-large-latest",     # [123B] top-tier reasoning model for high-complexity tasks.                            https://mistral.ai/news/mistral-large
    "mistral-medium-latest",    # [unspecified] frontier-class multimodal model released May 2025.                      https://mistral.ai/news/mistral-medium-3
    "mistral-small-latest",     # [24B] enterprise-grade small model with the first version released Feb. 2024          https://mistral.ai/news/mistral-small-3-1
    "mistral-saba-latest",      # [24B] model for languages from the Middle East and South Asia.                        https://mistral.ai/news/mistral-saba
    "open-mistral-nemo",        # [12B] multilingual open source model released July 2024.                              https://mistral.ai/news/mistral-nemo
]