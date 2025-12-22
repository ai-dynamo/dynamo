<?xml-stylesheet type="text/xsl" href="akamai-eccu-flush.xslt"?>
<!--
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<!--
  Akamai ECCU (Edge Content Control Utility) XML Generator

  This XSLT template generates an ECCU request XML for flushing Akamai cache.
  It takes a path parameter and recursively builds the directory structure
  needed for the cache invalidation request.

  Usage:
    xsltproc -stringparam target-path "path/to/flush" akamai-eccu-flush.xslt akamai-eccu-flush.xslt

  The output XML will contain nested match:recursive-dirs elements that
  represent the path hierarchy for cache invalidation.
-->
<xsl:stylesheet xmlns:match="x"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version="1.0"
                exclude-result-prefixes="match">

  <xsl:output method="xml" version="1.0" indent="yes"/>

  <!-- Parameter: The path to flush in Akamai cache -->
  <xsl:param name="target-path"/>

  <!-- Root template: Create the ECCU wrapper -->
  <xsl:template match="/">
    <eccu>
      <xsl:call-template name="dir">
        <xsl:with-param name="dir" select="$target-path"/>
      </xsl:call-template>
    </eccu>
  </xsl:template>

  <!-- Recursive template: Build nested directory structure -->
  <xsl:template name="dir">
    <xsl:param name="dir"/>

    <!-- Extract the first path segment before '/' -->
    <xsl:variable name="path" select="substring-before($dir, '/')"/>

    <!-- Get remaining path after the first '/' -->
    <xsl:variable name="rest" select="substring-after($dir, '/')"/>

    <xsl:choose>
      <!-- If there are more path segments, recurse -->
      <xsl:when test="string-length($rest) &gt; 0">
        <match:recursive-dirs value="{$path}">
          <xsl:call-template name="dir">
            <xsl:with-param name="dir" select="$rest"/>
          </xsl:call-template>
        </match:recursive-dirs>
      </xsl:when>

      <!-- Final path segment: add revalidate directive -->
      <xsl:otherwise>
        <match:recursive-dirs value="{$dir}">
          <revalidate>now</revalidate>
        </match:recursive-dirs>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

</xsl:stylesheet>
