import { GenerateOptions } from "../../knowledge/types.js";

export function renderDetails(options: GenerateOptions): string {
  const summary = options.label || (options.summary as string) || "Details";
  const content = (options.content as string) || "Content goes here.";

  return `<details class="govuk-details">
  <summary class="govuk-details__summary">
    <span class="govuk-details__summary-text">
      ${summary}
    </span>
  </summary>
  <div class="govuk-details__text">
    ${content}
  </div>
</details>`;
}
