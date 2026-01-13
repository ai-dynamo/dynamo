import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'NVIDIA Dynamo',
  tagline: 'High-performance, low-latency inference framework',
  favicon: 'img/favicon.ico',

  // Future flags
  future: {
    v4: true,
  },

  // For local testing
  url: 'http://localhost:3000',
  baseUrl: '/',

  organizationName: 'ai-dynamo',
  projectName: 'dynamo',

  onBrokenLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Enable Mermaid diagrams and configure markdown hooks
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
      onBrokenMarkdownImages: 'warn',
    },
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/', // Docs at root URL
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/ai-dynamo/dynamo/tree/main/docs/',
          showLastUpdateTime: true,
          // Versioning configuration - start fresh after restructure
          // Run `npm run docusaurus docs:version X.Y.Z` to create versions
        },
        blog: false, // Disable blog
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          // Preserve existing redirects from Sphinx structure
          {from: '/guides/tool-calling', to: '/agents/tool-calling'},
          {from: '/architecture/architecture', to: '/design_docs/architecture'},
          {from: '/architecture/disagg_serving', to: '/design_docs/disagg_serving'},
          {from: '/architecture/distributed_runtime', to: '/design_docs/distributed_runtime'},
          {from: '/architecture/dynamo_flow', to: '/design_docs/dynamo_flow'},
        ],
      },
    ],
    // Local search for offline/self-hosted use
    require.resolve('@easyops-cn/docusaurus-search-local'),
  ],

  themeConfig: {
    image: 'img/nvidia-social-card.png',
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'NVIDIA Dynamo',
      logo: {
        alt: 'NVIDIA Logo',
        src: 'img/nvidia-logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docs',
          position: 'left',
          label: 'Documentation',
        },
        {
          type: 'docsVersionDropdown',
          position: 'right',
          dropdownActiveClassDisabled: true,
        },
        {
          href: 'https://github.com/ai-dynamo/dynamo',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Getting Started', to: '/'},
            {label: 'Backends', to: '/backends/vllm/'},
            {label: 'Kubernetes', to: '/kubernetes/installation_guide'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub', href: 'https://github.com/ai-dynamo/dynamo'},
            {label: 'Issues', href: 'https://github.com/ai-dynamo/dynamo/issues'},
            {label: 'Discussions', href: 'https://github.com/ai-dynamo/dynamo/discussions'},
          ],
        },
        {
          title: 'NVIDIA',
          items: [
            {label: 'NVIDIA.com', href: 'https://www.nvidia.com'},
            {label: 'Privacy Policy', href: 'https://www.nvidia.com/en-us/about-nvidia/privacy-policy/'},
            {label: 'Terms of Service', href: 'https://www.nvidia.com/en-us/about-nvidia/terms-of-service/'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} NVIDIA Corporation & Affiliates. All rights reserved.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'yaml', 'rust', 'toml', 'json', 'docker'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
