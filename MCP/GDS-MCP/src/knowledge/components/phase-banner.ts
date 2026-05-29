import { ComponentDefinition } from "../types.js";

export const phaseBanner: ComponentDefinition = {
  slug: "phase-banner",
  name: "Phase banner",
  category: "layout",
  description:
    "A banner showing the service phase (alpha/beta) with a feedback link, placed below the header.",
  useCases: [
    "alpha or beta banner",
    "service phase indicator",
    "feedback link banner",
    "show service is in development",
  ],
  antiUseCases: [
    "Don't use for live services that have passed their beta assessment.",
  ],
  relatedComponents: ["tag", "header"],
  template: {
    baseMarkup: `<div class="govuk-phase-banner">
  <p class="govuk-phase-banner__content">
    <strong class="govuk-tag govuk-phase-banner__content__tag">
      {{phase}}
    </strong>
    <span class="govuk-phase-banner__text">
      This is a new service – your <a class="govuk-link" href="{{feedbackUrl}}">feedback</a> will help us to improve it.
    </span>
  </p>
</div>`,
    slots: [
      { name: "phase", required: false, type: "text", description: "Phase tag text (alpha or beta)", default: "beta" },
      { name: "feedbackUrl", required: false, type: "text", description: "URL for the feedback link", default: "/feedback" },
    ],
    variants: [],
  },
  accessibilityRules: [],
};
