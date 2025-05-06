#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

"""
User facing python APIs for managing local bentos and build new bentos.
"""

from __future__ import annotations

import json
import logging
import os
import typing as t

import fs
import fs.errors
import fs.mirror
import yaml
from bentoml._internal.bento.bento import BENTO_PROJECT_DIR_NAME, BENTO_README_FILENAME
from bentoml._internal.bento.bento import Bento as BaseBento
from bentoml._internal.bento.bento import (
    BentoApiInfo,
    BentoInfo,
    BentoInfoV2,
    BentoModelInfo,
    BentoRunnerInfo,
    BentoServiceInfo,
    get_default_svc_readme,
    get_service_import_str,
)
from bentoml._internal.bento.build_config import BentoBuildConfig, BentoPathSpec
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.service import Service
from bentoml._internal.service.loader import load
from bentoml._internal.tag import Tag, to_snake_case
from bentoml._internal.utils.filesystem import copy_file_to_fs_folder
from bentoml._internal.utils.uri import encode_path_for_uri
from bentoml.exceptions import BentoMLException, InvalidArgument
from fs.copy import copy_file
from fs.tempfs import TempFS
from simple_di import Provide, inject

from dynamo.sdk.lib.service import LinkedServices

logger = logging.getLogger(__name__)


class Bento(BaseBento):
    """Dynamo's Bento class that extends BentoML's Bento with additional functionality."""

    @classmethod
    @inject
    def create(
        cls,
        build_config: BentoBuildConfig,
        version: t.Optional[str] = None,
        build_ctx: t.Optional[str] = None,
        platform: t.Optional[str] = None,
        bare: bool = False,
        reload: bool = False,
        enabled_features: list[str] = Provide[BentoMLContainer.enabled_features],
    ) -> Bento:
        from _bentoml_sdk.images import Image, populate_image_from_build_config
        from _bentoml_sdk.models import BentoModel

        build_ctx = (
            os.getcwd()
            if build_ctx is None
            else os.path.realpath(os.path.expanduser(build_ctx))
        )
        if not os.path.isdir(build_ctx):
            raise InvalidArgument(
                f"Bento build context {build_ctx} does not exist or is not a directory."
            )

        BentoMLContainer.model_aliases.set(build_config.model_aliases)
        # This also verifies that svc can be imported correctly
        svc = load(build_config.service, working_dir=build_ctx, reload=reload)

        # TODO: At some point we need this to take place within the load function
        LinkedServices.remove_unused_edges()

        if not build_config.service:
            object.__setattr__(build_config, "service", get_service_import_str(svc))
        is_legacy = isinstance(svc, Service)
        # Apply default build options
        image: Image | None = None
        disable_image = "no_image" in enabled_features or is_legacy

        if isinstance(svc, Service):
            # for < 1.2
            bento_name = (
                build_config.name if build_config.name is not None else svc.name
            )
        else:
            # for >= 1.2
            svc.inject_config()
            bento_name = (
                build_config.name
                if build_config.name is not None
                else to_snake_case(svc.name)
            )
            build_config.envs.extend(svc.envs)
            build_config.labels.update(svc.labels)
            if svc.image is not None:
                image = Image(base_image=svc.image)
        if not disable_image:
            image = populate_image_from_build_config(image, build_config, build_ctx)
        build_config = build_config.with_defaults()
        tag = Tag(bento_name, version)
        if version is None:
            tag = tag.make_new_version()

        logger.debug(
            'Building BentoML service "%s" from build context "%s".', tag, build_ctx
        )
        bento_fs = TempFS(
            identifier=f"bentoml_bento_{bento_name}",
            temp_dir=BentoMLContainer.tmp_bento_store_dir.get(),
        )
        models: list[BentoModelInfo] = []

        def append_model(model: BentoModelInfo) -> None:
            if model not in models:
                models.append(model)

        if build_config.models:
            for model_spec in build_config.models:
                model = BentoModel(model_spec.tag)
                append_model(model.to_info(model_spec.alias))
        elif is_legacy:
            # XXX: legacy way to get models from service
            # Add all models required by the service
            for model in svc.models:
                append_model(BentoModel(model.tag).to_info())
            # Add all models required by service runners
            for runner in svc.runners:
                for model in runner.models:
                    append_model(BentoModel(model.tag).to_info())

        if not bare:
            ctx_fs = fs.open_fs(encode_path_for_uri(build_ctx))

            # create ignore specs
            specs = BentoPathSpec(build_config.include, build_config.exclude, build_ctx)

            # Copy all files base on include and exclude, into `src` directory
            relpaths = [s for s in build_config.include if s.startswith("../")]
            if len(relpaths) != 0:
                raise InvalidArgument(
                    "Paths outside of the build context directory cannot be included; use a symlink or copy those files into the working directory manually."
                )
            bento_fs.makedir(BENTO_PROJECT_DIR_NAME)
            target_fs = bento_fs.opendir(BENTO_PROJECT_DIR_NAME)
            with target_fs.open("bentofile.yaml", "w") as bentofile_yaml:
                build_config.to_yaml(bentofile_yaml)

            for dir_path, _, files in ctx_fs.walk():
                for f in files:
                    path = fs.path.combine(dir_path, f.name).lstrip("/")
                    if specs.includes(path):
                        if ctx_fs.getsize(path) > 10 * 1024 * 1024:
                            logger.warning("File size is larger than 10MiB: %s", path)
                        target_fs.makedirs(dir_path, recreate=True)
                        copy_file(ctx_fs, path, target_fs, path)
            if image is None:
                # NOTE: we need to generate both Python and Conda
                # first to make sure we can generate the Dockerfile correctly.
                build_config.python.write_to_bento(
                    bento_fs, build_ctx, platform_=platform
                )
                build_config.conda.write_to_bento(bento_fs, build_ctx)
                build_config.docker.write_to_bento(
                    bento_fs, build_ctx, build_config.conda
                )

            # Create `readme.md` file
            if (
                build_config.description is not None
                and build_config.description.startswith("file:")
            ):
                file_name = build_config.description[5:].strip()
                if not ctx_fs.exists(file_name):
                    raise InvalidArgument(f"File {file_name} does not exist.")
                copy_file_to_fs_folder(
                    ctx_fs.getsyspath(file_name),
                    bento_fs,
                    dst_filename=BENTO_README_FILENAME,
                )
            elif build_config.description is None and ctx_fs.exists(
                BENTO_README_FILENAME
            ):
                copy_file_to_fs_folder(
                    ctx_fs.getsyspath(BENTO_README_FILENAME),
                    bento_fs,
                    dst_filename=BENTO_README_FILENAME,
                )
            else:
                with bento_fs.open(BENTO_README_FILENAME, "w") as f:
                    if build_config.description is None:
                        f.write(get_default_svc_readme(svc, version))
                    else:
                        f.write(build_config.description)

            # Create 'apis/openapi.yaml' file
            bento_fs.makedir("apis")
            with bento_fs.open(fs.path.combine("apis", "openapi.yaml"), "w") as f:
                yaml.dump(svc.openapi_spec, f)
            if not is_legacy:
                with bento_fs.open(fs.path.combine("apis", "schema.json"), "w") as f:
                    json.dump(svc.schema(), f, indent=2)

        if image is None:
            bento_info = BentoInfo(
                tag=tag,
                service=svc,  # type: ignore # attrs converters do not typecheck
                entry_service=svc.name,
                labels=build_config.labels,
                models=models,
                runners=(
                    [BentoRunnerInfo.from_runner(r) for r in svc.runners]  # type: ignore # attrs converters do not typecheck
                    if is_legacy
                    else []
                ),
                apis=(
                    [BentoApiInfo.from_inference_api(api) for api in svc.apis.values()]
                    if is_legacy
                    else []
                ),
                services=(
                    [
                        BentoServiceInfo.from_service(s)
                        for s in svc.all_services().values()
                    ]
                    if not is_legacy
                    else []
                ),
                docker=build_config.docker,
                python=build_config.python,
                conda=build_config.conda,
                envs=build_config.envs,
                schema=svc.schema() if not is_legacy else {},
            )
        else:
            bento_info = BentoInfoV2(
                tag=tag,
                service=svc,  # type: ignore # attrs converters do not typecheck
                entry_service=svc.name,
                labels=build_config.labels,
                models=models,
                services=(
                    [
                        BentoServiceInfo.from_service(s)
                        for s in svc.all_services().values()
                    ]
                    if not is_legacy
                    else []
                ),
                envs=build_config.envs,
                schema=svc.schema() if not is_legacy else {},
                image=image.freeze(bento_fs, build_config.envs, platform),
            )

        res = Bento(tag, bento_fs, bento_info)
        if bare:
            return res
        # Create bento.yaml
        res.flush_info()
        try:
            res.validate()
        except BentoMLException as e:
            raise BentoMLException(f"Failed to create {res!s}: {e}") from None

        return res
