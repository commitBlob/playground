import { ComponentDefinition } from "../types.js";

export const skipLink: ComponentDefinition = {
  slug: "skip-link",
  name: "Skip link",
  category: "navigation",
  description:
    "A hidden link that becomes visible on focus, allowing keyboard users to skip navigation and go straight to page content.",
  useCases: [
    "skip to main content",
    "keyboard navigation shortcut",
    "skip navigation for screen readers",
    "accessibility skip link",
  ],
  antiUseCases: [
    "Every page should have a skip link — there's no reason not to use it.",
  ],
  relatedComponents: [],
  template: {
    baseMarkup: `<a href="#main-content" class="govuk-skip-link" data-module="govuk-skip-link">Skip to main content</a>`,
    slots: [
      { name: "href", required: false, type: "text", description: "Target id (default: #main-content)", default: "#main-content" },
    ],
    variants: [],
  },
  accessibilityRules: [
    {
      id: "skip-link-target-exists",
      severity: "warning",
      description: "Skip link href target should exist on the page",
      check: (html) => {
        if (!/govuk-skip-link/.test(html)) return { passed: true, message: "No skip link present." };
        const hrefMatch = html.match(/govuk-skip-link[^>]*href=["']#([^"']+)["']/);
        if (hrefMatch) {
          const targetId = hrefMatch[1];
          if (!new RegExp(`id=["']${targetId}["']`).test(html)) {
            return { passed: false, message: `Skip link targets #${targetId} but no element with that id found.`, suggestion: `Add id="${targetId}" to the main content area.` };
          }
        }
        return { passed: true, message: "Skip link target check passed." };
      },
    },
  ],
};
