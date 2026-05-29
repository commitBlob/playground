import { GenerateOptions } from "../../knowledge/types.js";

export function renderPhaseBanner(options: GenerateOptions): string {
  const phase = (options.phase as string) || options.variant || "beta";
  const feedbackUrl = (options.feedbackUrl as string) || "/feedback";

  return `<div class="govuk-phase-banner">
  <p class="govuk-phase-banner__content">
    <strong class="govuk-tag govuk-phase-banner__content__tag">
      ${phase}
    </strong>
    <span class="govuk-phase-banner__text">
      This is a new service – your <a class="govuk-link" href="${feedbackUrl}">feedback</a> will help us to improve it.
    </span>
  </p>
</div>`;
}
