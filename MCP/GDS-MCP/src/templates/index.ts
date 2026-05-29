import { ComponentDefinition, GenerateOptions } from "../knowledge/types.js";
import { renderTextInput } from "./components/text-input.js";
import { renderTextarea } from "./components/textarea.js";
import { renderCheckboxes } from "./components/checkboxes.js";
import { renderRadios } from "./components/radios.js";
import { renderCharacterCount } from "./components/character-count.js";
import { renderDateInput } from "./components/date-input.js";
import { renderFileUpload } from "./components/file-upload.js";
import { renderPasswordInput } from "./components/password-input.js";
import { renderSelect } from "./components/select.js";
import { renderAccordion } from "./components/accordion.js";
import { renderBackLink } from "./components/back-link.js";
import { renderBreadcrumbs } from "./components/breadcrumbs.js";
import { renderPagination } from "./components/pagination.js";
import { renderSkipLink } from "./components/skip-link.js";
import { renderTabs } from "./components/tabs.js";
import { renderDetails } from "./components/details.js";
import { renderFieldset } from "./components/fieldset.js";
import { renderInsetText } from "./components/inset-text.js";
import { renderSummaryList } from "./components/summary-list.js";
import { renderTable } from "./components/table.js";
import { renderTaskList } from "./components/task-list.js";
import { renderTag } from "./components/tag.js";
import { renderErrorMessage } from "./components/error-message.js";
import { renderErrorSummary } from "./components/error-summary.js";
import { renderNotificationBanner } from "./components/notification-banner.js";
import { renderPanel } from "./components/panel.js";
import { renderWarningText } from "./components/warning-text.js";
import { renderButton } from "./components/button.js";
import { renderHeader } from "./components/header.js";
import { renderFooter } from "./components/footer.js";
import { renderCookieBanner } from "./components/cookie-banner.js";
import { renderExitThisPage } from "./components/exit-this-page.js";
import { renderPhaseBanner } from "./components/phase-banner.js";
import { renderServiceNavigation } from "./components/service-navigation.js";

export function renderComponent(
  component: ComponentDefinition,
  options: GenerateOptions
): string {
  switch (component.slug) {
    case "text-input":
      return renderTextInput(options);
    case "textarea":
      return renderTextarea(options);
    case "checkboxes":
      return renderCheckboxes(options);
    case "radios":
      return renderRadios(options);
    case "character-count":
      return renderCharacterCount(options);
    case "date-input":
      return renderDateInput(options);
    case "file-upload":
      return renderFileUpload(options);
    case "password-input":
      return renderPasswordInput(options);
    case "select":
      return renderSelect(options);
    case "accordion":
      return renderAccordion(options);
    case "back-link":
      return renderBackLink(options);
    case "breadcrumbs":
      return renderBreadcrumbs(options);
    case "pagination":
      return renderPagination(options);
    case "skip-link":
      return renderSkipLink(options);
    case "tabs":
      return renderTabs(options);
    case "details":
      return renderDetails(options);
    case "fieldset":
      return renderFieldset(options);
    case "inset-text":
      return renderInsetText(options);
    case "summary-list":
      return renderSummaryList(options);
    case "table":
      return renderTable(options);
    case "task-list":
      return renderTaskList(options);
    case "tag":
      return renderTag(options);
    case "error-message":
      return renderErrorMessage(options);
    case "error-summary":
      return renderErrorSummary(options);
    case "notification-banner":
      return renderNotificationBanner(options);
    case "panel":
      return renderPanel(options);
    case "warning-text":
      return renderWarningText(options);
    case "button":
      return renderButton(options);
    case "header":
      return renderHeader(options);
    case "footer":
      return renderFooter(options);
    case "cookie-banner":
      return renderCookieBanner(options);
    case "exit-this-page":
      return renderExitThisPage(options);
    case "phase-banner":
      return renderPhaseBanner(options);
    case "service-navigation":
      return renderServiceNavigation(options);
    default:
      throw new Error(
        `No template renderer implemented for component slug: "${component.slug}"`
      );
  }
}
