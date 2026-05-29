import { ComponentDefinition } from "../types.js";

export const pagination: ComponentDefinition = {
  slug: "pagination",
  name: "Pagination",
  category: "navigation",
  description:
    "Navigate between pages of results or content, with previous/next links and page numbers.",
  useCases: [
    "navigate between pages",
    "page numbers",
    "next and previous links",
    "paginated results",
    "multi-page content",
  ],
  antiUseCases: [
    "If content fits on one page, don't paginate.",
    "For step-by-step navigation within a form, use separate pages without pagination.",
  ],
  relatedComponents: ["breadcrumbs"],
  template: {
    baseMarkup: `<nav class="govuk-pagination" aria-label="Pagination">
  <div class="govuk-pagination__prev">
    <a class="govuk-link govuk-pagination__link" href="{{prevHref}}" rel="prev">
      <svg class="govuk-pagination__icon govuk-pagination__icon--prev" xmlns="http://www.w3.org/2000/svg" height="13" width="15" aria-hidden="true" focusable="false" viewBox="0 0 15 13">
        <path d="m6.5938-0.0078125-6.7266 6.7266 6.7441 6.4062 1.377-1.449-4.1856-3.9768h12.896v-2h-12.984l4.2931-4.293-1.414-1.414z"></path>
      </svg>
      <span class="govuk-pagination__link-title">
        Previous<span class="govuk-visually-hidden"> page</span>
      </span>
    </a>
  </div>
  <ul class="govuk-pagination__list">
{{pageItems}}
  </ul>
  <div class="govuk-pagination__next">
    <a class="govuk-link govuk-pagination__link" href="{{nextHref}}" rel="next">
      <span class="govuk-pagination__link-title">
        Next<span class="govuk-visually-hidden"> page</span>
      </span>
      <svg class="govuk-pagination__icon govuk-pagination__icon--next" xmlns="http://www.w3.org/2000/svg" height="13" width="15" aria-hidden="true" focusable="false" viewBox="0 0 15 13">
        <path d="m8.107-0.0078125-1.4136 1.414 4.2926 4.293h-12.986v2h12.896l-4.1855 3.9766 1.377 1.4492 6.7441-6.4062-6.7246-6.7266z"></path>
      </svg>
    </a>
  </div>
</nav>`,
    slots: [
      { name: "prevHref", required: false, type: "text", description: "URL for previous page" },
      { name: "nextHref", required: false, type: "text", description: "URL for next page" },
      { name: "items", required: false, type: "array", description: "Page number items" },
    ],
    variants: [
      { name: "block", description: "Only prev/next links without page numbers (for content pages)" },
    ],
  },
  accessibilityRules: [
    {
      id: "pagination-aria-label",
      severity: "error",
      description: "Pagination nav must have aria-label",
      check: (html) => {
        if (!/govuk-pagination/.test(html)) return { passed: true, message: "No pagination present." };
        if (!/aria-label/.test(html)) {
          return { passed: false, message: "Pagination <nav> missing aria-label.", suggestion: "Add aria-label=\"Pagination\" to the nav element." };
        }
        return { passed: true, message: "Pagination has aria-label." };
      },
    },
  ],
};
