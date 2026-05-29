import { ComponentDefinition } from "../types.js";
import { textInput } from "./text-input.js";
import { textarea } from "./textarea.js";
import { checkboxes } from "./checkboxes.js";
import { radios } from "./radios.js";
import { characterCount } from "./character-count.js";
import { dateInput } from "./date-input.js";
import { fileUpload } from "./file-upload.js";
import { passwordInput } from "./password-input.js";
import { select } from "./select.js";
import { accordion } from "./accordion.js";
import { backLink } from "./back-link.js";
import { breadcrumbs } from "./breadcrumbs.js";
import { pagination } from "./pagination.js";
import { skipLink } from "./skip-link.js";
import { serviceNavigation } from "./service-navigation.js";
import { tabs } from "./tabs.js";
import { details } from "./details.js";
import { fieldset } from "./fieldset.js";
import { insetText } from "./inset-text.js";
import { summaryList } from "./summary-list.js";
import { table } from "./table.js";
import { taskList } from "./task-list.js";
import { tag } from "./tag.js";
import { errorMessage } from "./error-message.js";
import { errorSummary } from "./error-summary.js";
import { notificationBanner } from "./notification-banner.js";
import { panel } from "./panel.js";
import { warningText } from "./warning-text.js";
import { button } from "./button.js";
import { header } from "./header.js";
import { footer } from "./footer.js";
import { cookieBanner } from "./cookie-banner.js";
import { exitThisPage } from "./exit-this-page.js";
import { phaseBanner } from "./phase-banner.js";

const components: ComponentDefinition[] = [
  textInput,
  textarea,
  checkboxes,
  radios,
  characterCount,
  dateInput,
  fileUpload,
  passwordInput,
  select,
  accordion,
  backLink,
  breadcrumbs,
  pagination,
  skipLink,
  serviceNavigation,
  tabs,
  details,
  fieldset,
  insetText,
  summaryList,
  table,
  taskList,
  tag,
  errorMessage,
  errorSummary,
  notificationBanner,
  panel,
  warningText,
  button,
  header,
  footer,
  cookieBanner,
  exitThisPage,
  phaseBanner,
];

export function getAllComponents(): ComponentDefinition[] {
  return components;
}

export function getComponentBySlug(
  slug: string
): ComponentDefinition | undefined {
  return components.find((c) => c.slug === slug);
}
