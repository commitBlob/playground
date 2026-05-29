import { GenerateOptions } from "../../knowledge/types.js";

export function renderPanel(options: GenerateOptions): string {
  const title = options.label || (options.title as string) || "Application complete";
  const body = (options.content as string) || (options.body as string) || "";

  let html = `<div class="govuk-panel govuk-panel--confirmation">
  <h1 class="govuk-panel__title">
    ${title}
  </h1>`;

  if (body) {
    html += `\n  <div class="govuk-panel__body">\n    ${body}\n  </div>`;
  }

  html += `\n</div>`;
  return html;
}
