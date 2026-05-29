import { GenerateOptions } from "../../knowledge/types.js";
import { toSlug } from "../helpers.js";

export function renderTaskList(options: GenerateOptions): string {
  const items = options.items || [];

  const tasksHtml = items
    .map((item) => {
      const status = item.hint || "Not yet started";
      const tagClass = item.checked ? "govuk-tag" : "govuk-tag govuk-tag--blue";
      return `  <li class="govuk-task-list__item govuk-task-list__item--with-link">
    <div class="govuk-task-list__name-and-hint">
      <a class="govuk-link govuk-task-list__link" href="${item.value}" aria-describedby="task-${toSlug(item.text)}-status">
        ${item.text}
      </a>
    </div>
    <div class="govuk-task-list__status" id="task-${toSlug(item.text)}-status">
      <strong class="${tagClass}">${status}</strong>
    </div>
  </li>`;
    })
    .join("\n");

  return `<ul class="govuk-task-list">
${tasksHtml}
</ul>`;
}
