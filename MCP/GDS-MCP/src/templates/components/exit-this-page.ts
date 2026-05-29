import { GenerateOptions } from "../../knowledge/types.js";

export function renderExitThisPage(options: GenerateOptions): string {
  const href = (options.href as string) || "https://www.bbc.co.uk/weather";

  return `<div class="govuk-exit-this-page" data-module="govuk-exit-this-page">
  <a href="${href}" role="button" draggable="false" class="govuk-button govuk-button--warning govuk-exit-this-page__button govuk-js-exit-this-page-button" data-module="govuk-button">
    <span class="govuk-visually-hidden">Emergency</span> Exit this page
  </a>
</div>`;
}
