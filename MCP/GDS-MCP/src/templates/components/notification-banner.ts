import { GenerateOptions } from "../../knowledge/types.js";

export function renderNotificationBanner(options: GenerateOptions): string {
  const title = (options.title as string) || "Important";
  const heading = options.label || (options.heading as string) || "Notification message";
  const isSuccess = options.variant === "success" || (options.type as string) === "success";
  const role = isSuccess ? "alert" : "region";
  const successClass = isSuccess ? " govuk-notification-banner--success" : "";

  return `<div class="govuk-notification-banner${successClass}" role="${role}" aria-labelledby="govuk-notification-banner-title" data-module="govuk-notification-banner">
  <div class="govuk-notification-banner__header">
    <h2 class="govuk-notification-banner__title" id="govuk-notification-banner-title">
      ${isSuccess ? "Success" : title}
    </h2>
  </div>
  <div class="govuk-notification-banner__content">
    <p class="govuk-notification-banner__heading">
      ${heading}
    </p>
  </div>
</div>`;
}
