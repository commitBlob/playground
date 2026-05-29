import { GenerateOptions } from "../../knowledge/types.js";

export function renderBackLink(options: GenerateOptions): string {
  const href = (options.href as string) || "/";
  return `<a href="${href}" class="govuk-back-link">Back</a>`;
}
