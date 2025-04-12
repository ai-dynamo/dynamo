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

import logging
import typing as t

import attr
from bentoml._internal.bento.bento import DEFAULT_BENTO_BUILD_FILES
from dynamo.sdk.lib.bento import Bento
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.tag import Tag
from bentoml._internal.utils.args import set_arguments
from bentoml._internal.utils.filesystem import resolve_user_filepath
from bentoml.exceptions import InvalidArgument
from simple_di import Provide, inject

if t.TYPE_CHECKING:
    from bentoml._internal.bento.bento import BentoStore

logger = logging.getLogger(__name__)


@inject
def import_bento(
    path: str,
    input_format: str | None = None,
    *,
    protocol: str | None = None,
    user: str | None = None,
    passwd: str | None = None,
    params: t.Optional[t.Dict[str, str]] = None,
    subpath: str | None = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> Bento:
    """
    Import a bento.

    Examples:

    .. code-block:: python

        # imports 'my_bento' from '/path/to/folder/my_bento.bento'
        bentoml.import_bento('/path/to/folder/my_bento.bento')

        # imports 'my_bento' from '/path/to/folder/my_bento.tar.gz'
        # currently supported formats are tar.gz ('gz'),
        # tar.xz ('xz'), tar.bz2 ('bz2'), and zip
        bentoml.import_bento('/path/to/folder/my_bento.tar.gz')
        # treats 'my_bento.ext' as a gzipped tarfile
        bentoml.import_bento('/path/to/folder/my_bento.ext', 'gz')

        # imports 'my_bento', which is stored as an
        # uncompressed folder, from '/path/to/folder/my_bento/'
        bentoml.import_bento('/path/to/folder/my_bento', 'folder')

        # imports 'my_bento' from the S3 bucket 'my_bucket',
        # path 'folder/my_bento.bento'
        # requires `fs-s3fs <https://pypi.org/project/fs-s3fs/>`_
        bentoml.import_bento('s3://my_bucket/folder/my_bento.bento')
        bentoml.import_bento('my_bucket/folder/my_bento.bento', protocol='s3')
        bentoml.import_bento('my_bucket', protocol='s3',
                             subpath='folder/my_bento.bento')
        bentoml.import_bento('my_bucket', protocol='s3',
                             subpath='folder/my_bento.bento',
                             user='<AWS access key>', passwd='<AWS secret key>',
                             params={'acl': 'public-read',
                                     'cache-control': 'max-age=2592000,public'})

    For a more comprehensive description of what each of the keyword arguments
    (:code:`protocol`, :code:`user`, :code:`passwd`,
     :code:`params`, and :code:`subpath`) mean, see the
    `FS URL documentation <https://docs.pyfilesystem.org/en/latest/openers.html>`_.

    Args:
        tag: the tag of the bento to export
        path: can be one of two things:
              * a folder on the local filesystem
              * an `FS URL <https://docs.pyfilesystem.org/en/latest/openers.html>`_,
                for example :code:`'s3://my_bucket/folder/my_bento.bento'`
        protocol: (expert) The FS protocol to use when exporting. Some example protocols
                  are :code:`'ftp'`, :code:`'s3'`, and :code:`'userdata'`
        user: (expert) the username used for authentication if required, e.g. for FTP
        passwd: (expert) the username used for authentication if required, e.g. for FTP
        params: (expert) a map of parameters to be passed to the FS used for
                export, e.g. :code:`{'proxy': 'myproxy.net'}` for setting a
                proxy for FTP
        subpath: (expert) the path inside the FS that the bento should be exported to
        _bento_store: the bento store to save the bento to

    Returns:
        Bento: the imported bento
    """
    return Bento.import_from(
        path,
        input_format,
        protocol=protocol,
        user=user,
        passwd=passwd,
        params=params,
        subpath=subpath,
    ).save(_bento_store)

@inject
def build_bentofile(
    bentofile: str | None = None,
    *,
    service: str | None = None,
    name: str | None = None,
    version: str | None = None,
    labels: dict[str, str] | None = None,
    build_ctx: str | None = None,
    platform: str | None = None,
    bare: bool = False,
    reload: bool = False,
    args: dict[str, t.Any] | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
) -> Bento:
    """
    Build a Bento base on options specified in a bentofile.yaml file.

    By default, this function will look for a `bentofile.yaml` file in current working
    directory.

    Args:
        bentofile: The file path to build config yaml file
        version: Override the default auto generated version str
        build_ctx: Build context directory, when used as
        bare: whether to build a bento without copying files
        reload: whether to reload the service

    Returns:
        Bento: a Bento instance representing the materialized Bento saved in BentoStore
    """
    if args is not None:
        set_arguments(**args)
    if bentofile:
        try:
            bentofile = resolve_user_filepath(bentofile, None)
        except FileNotFoundError:
            raise InvalidArgument(f'bentofile "{bentofile}" not found')
        else:
            build_config = BentoBuildConfig.from_file(bentofile)
    else:
        for filename in DEFAULT_BENTO_BUILD_FILES:
            try:
                bentofile = resolve_user_filepath(filename, build_ctx)
            except FileNotFoundError:
                pass
            else:
                build_config = BentoBuildConfig.from_file(bentofile)
                break
        else:
            build_config = BentoBuildConfig(service=service or "")

    new_attrs = {}
    if name is not None:
        new_attrs["name"] = name
    if labels:
        new_attrs["labels"] = {**(build_config.labels or {}), **labels}

    if new_attrs:
        build_config = attr.evolve(build_config, **new_attrs)

    bento = Bento.create(
        build_config=build_config,
        version=version,
        build_ctx=build_ctx,
        platform=platform,
        bare=bare,
        reload=reload,
    )
    if not bare:
        return bento.save(_bento_store)
    return bento
