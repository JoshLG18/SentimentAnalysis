var AOP = AOP || {};
var modalSelectorString = 'a.delete-modal, a[data-reveal-id], a[class*=export-citation], a[class*=ip-address]';
var modalFocusableSelectorString = 'a[href], area[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), button:not([disabled]), [tabindex="0"]';

var qtipSelectorString = 'a[classG*="hasTooltipCustom"], a[data-hasqtip], a[data-qtip-event]';

var tooltipIconSelectorString = '.tooltip-icon';

var imagePath = AOP.baseUrl + '/cambridge-core/public/images/';
var enterKey = 13;

AOP.toolTipCloseLinkHtml = '<p> \
  <a style="float: right; color:#0072cf;" class="button small transparent-no-border radius tooltip-close-btn" \
  id="tooltip-close-link" href="#" aria-label="Close help"> \
    <span class="custom-tooltip-button-remove"> \
      <img class="icon-svg" src="' + imagePath + 'btn_item_remove.svg" alt="Close tooltip"/> \
    </span> \
  </a> \
</p>';

AOP.lastElementBeforeRevealModal = '';

AOP.enableKeyboardAccessInQtipTooltip = function () {
  setTimeout(function () {
    AOP.enableKeyboardAccessOnQtip($('.qtip-content'));
  }, 300);
};

AOP.attatchCloseLinkToQtip = function (qtipAPI) {
  setTimeout(function () {
    var closeLink = $('.qtip-content').find('a[id^=tooltip-close-link]');
    if (closeLink.length === 0) {
      return false;
    }
    closeLink.on('click', function (e) {
      e.preventDefault();
      qtipAPI.toggle(false);
      $(AOP.lastElementBeforeRevealModal).focus();
    });
    closeLink.on('keydown', function (e) {
      var keyPressed = e.keyCode || e.which;

      if (keyPressed === enterKey) {
        e.preventDefault();
        qtipAPI.toggle(false);
        $(AOP.lastElementBeforeRevealModal).focus();
      }
    });
  }, 300);
};

/* This relies on the behaviour set by AOP.attatchCloseLinkToQtip
 * This will simulate a click on the close button when the escape key
 * is pressed.
 */
AOP.closeQtipOnEsc = function (qtipAPI) {
  setTimeout(function() {
    var closeLink = $('.qtip-content').find('a[id^=tooltip-close-link]');
    var qtip = $('.qtip-content');
    qtip.on('keydown', function (e) {
      var keyPressed = e.keyCode || e.which;
      var escKey = 27;

      if (keyPressed === escKey) {
        closeLink.click();
      }
    });

  }, 300);
};

AOP.enableKeyboardAccessOnQtip = function ($DOMElement) {
  var tabAbles = $DOMElement.find('select:visible, input:visible, textarea:visible, button:visible, a.button:visible, a.button-no-transition:visible').filter(function (index, element) {
    var $element = $(element);
    var elementClass = $element.attr('class');
    var elementType = $element.attr('type');
    var isHidden = elementType && elementType.split(' ').indexOf('hidden') > -1;
    return isHidden !== true;
  });

  var firstTabAble = tabAbles.first();
  var lastTabAble = tabAbles.last();

  if (tabAbles.length > 1) {
    var firstNonCloseTabable = $(tabAbles[0]).hasClass('tooltip-close-btn') ? tabAbles[1] : tabAbles[0];
  } else {
    var firstNonCloseTabable = tabAbles[0];
  }

  $(firstNonCloseTabable).focus();

  lastTabAble.on('keydown', function (e) {
    var keyPressed = e.keyCode || e.which;
    if (keyPressed === 9 && !e.shiftKey) {
      e.preventDefault();
      var firstTabAbleDisplayed = firstTabAble.is(':visible');
      if (firstTabAbleDisplayed) {
        firstTabAble.focus();
        return true;
      }
      if (tabAbles.length >= 2) {
        tabAbles[1].focus();
      }
      return true;
    }
  });
};

$(document).on('keydown', qtipSelectorString + ', ' + modalSelectorString, function (e) {
  var keyPressed = e.keyCode || e.which;
  if (keyPressed === enterKey) {
    AOP.lastElementBeforeRevealModal = e.target;
  }
});

/**
 * For the give switch, toggle the aria-label text, based on switch state.
 * @param $switch
 */
AOP.toggleSwitchAriaLabel = function($switch) {
  if (typeof $switch !== 'object') {
    return;
  }
  var dataOnMessage = $switch.data('onMessage');
  var dataOffMessage = $switch.data('offMessage');
  $switch.attr('aria-label', $switch.hasClass('off') ? dataOnMessage : dataOffMessage);
};

/**
 * Remove role alert from the label then add it to a div container
 * @param $parent
 */
AOP.roleAlertFn = function($parent) {
  $parent
  .find('input, textarea, select')
  .not('[data-abide-ignore], [type=checkbox]')
  .on('invalid.fndtn.abide valid.fndtn.abide', function(e) {
    e.preventDefault();

    var $el = $(this);
    var invalidField = $el.is('[data-invalid]:visible');
    var $el_parent = $el.parent();
    var parent = $el_parent.is('label') ? $el_parent.parent() : $el_parent;
    var label = parent.find('label[for="' + $el.attr('id') + '"]');
    var errorEl = parent.find('small.error');
    var errorContainer = parent.find('#error-container');
    var countryPicker = $('.country-picker .input-container input');

    if ($el[0] === countryPicker[0] && countryPicker.val() === '') {
      var errorMessageP = $('#countryPickerErrorMsg');
      errorMessageP.show();
      countryPicker.attr('aria-describedby', 'countryPickerErrorMsg');
    }

    if (label.length) {
      label.removeAttr('role');
    }

    if (errorEl.length) {
      if (!errorContainer.length && invalidField) {
        return errorEl.wrap('<div id="error-container" role="alert"></div>')
      }
      return invalidField ? errorContainer.show() : errorContainer.hide();
    }
  });
};

$(document).ready(function () {
  // On switch click/keydown, toggle the aria-label.
  $('.switch-wrapper').on('click', function (e) {
    e.preventDefault();
    var disableAriaToggle = $(this).data('disableAriaToggle');
    // Check if this action is disabled on the element - sometimes we want to manually control when this happens.
    if (disableAriaToggle !== true) {
      AOP.toggleSwitchAriaLabel($(this));
    }
  });

  // Override form validation
  AOP.roleAlertFn($('form'));

  //Insert tabindex on tabpanels
  $('.page-tabs a[role="tab"]').on('keyup', function () {
    $("div[role='tabpanel']").attr('tabindex', '0');
  });

  $('input[type="checkbox"]').on('keypress', function () {
    $(this).click();
  });

  var toolTipIconElement = $(tooltipIconSelectorString);

  if (toolTipIconElement.length > 0) {
    toolTipIconElement.each(function () {
      AOP.focusTrap($(this).attr('data-dropdown'));
    });
  }
});

// Enable keyboard access on tooltip modal
$(document).on('keydown', tooltipIconSelectorString, function (e) {
  var self = this;
  $(self).focus();

  var keyPressed = e.keyCode || e.which;
  if (keyPressed === enterKey) {
    AOP.lastElementBeforeRevealModal = self;
    setTimeout(function () {
      AOP.enableKeyboardAccess($('#' + $(self).attr('aria-controls')));
    }, 300);
  }
});

$(document).on('click', 'a[id^=tooltip-close-link]', function (e) {
  e.preventDefault();
  $(AOP.lastElementBeforeRevealModal).focus();
  return;
});

/* Start - Accessible Modals */

// Prevent mouse clicks outside modal
$(document).foundation({ 'reveal': { close_on_background_click: false } });

$(document).on('closed.fndtn.reveal', '[data-reveal], [data-reveal-id]', function () {
  $(AOP.lastElementBeforeRevealModal).focus();
  AOP.lastElementBeforeRevealModal = null;
});

$(document).on('opened.fndtn.reveal', '[data-reveal], [data-reveal-id]', function () {
  if (!AOP.lastElementBeforeRevealModal) {
    AOP.lastElementBeforeRevealModal = document.activeElement;
  }

  // if loader is visible, then don't call enableKeyboardAccess when modal has shown, instead, call manually after ajax call
  if (!$(this).find('div.loader:visible').length) {
    AOP.enableKeyboardAccess($(this));
  }
});

$(document).on('focus', 'input[type="checkbox"]', function (e) {
    $(e.target).focus(function () {
      $(this).next('span').css({'box-shadow': '0 0 0 2px #649FF9', 'border-radius': '3px'});
    });
    $(e.target).blur(function () {
      $(this).next('span').css('box-shadow', 'none');
    });
});


/**
 * Toggle tab index for all tabs
 * @param $target
 */
AOP.toggleTabIndex = function ($target) {
  if (!$target) return;
  var $parent = $target.parent();
  var $inactiveTabs = $parent.parent().find('.tab-title > [role="tab"]');

  $inactiveTabs
  .attr({
    'tabindex' : '-1',
    'aria-selected' : null
  });
  $target.attr({
    'aria-selected' : true
  });
}

/**
 * Toggle through tabs using arrow keys
 */
AOP.toggleArrowKeysOnTabs = function(e) {
  var $parent = $(e.target).parent();
  var tabRole = '[role="tab"]';
  var $prev = $parent.prev().children(tabRole);
  var $next = $parent.next().children(tabRole);
  var $target;

  switch (e.code) {
    case 'ArrowLeft':
      $target = $prev;
      break;
    case 'ArrowRight':
      $target = $next;
      break;
    default:
      $target = false;
      break;
  }

  if ($target && $target.length) {
    $target.focus();
    $target.on('keydown', function (e) {
      if (e.code === 'Enter') {
        AOP.toggleTabIndex($target);
      }
    });
  }
};

AOP.handleAltmetricsAccessibility = function () {
  $('div.altmetric-embed').on('altmetric:show', function () {
    var productSection = $(this).closest('.product-section');
    var productTitle = productSection.attr('data-test-title');
    var seeMoreDetails = $('p.altmetric-see-more-details');

    if (seeMoreDetails) {
      modifySeeMoreDetails();
    }
    modifyLogoDetails(productTitle, this);
  });
}

function modifySeeMoreDetails () {
  var seeMoreDetailsLink = $('p.altmetric-see-more-details a');
  var seeMoreDetailsSpan = document.createElement('span');
  // '\u00A0' is for whitespace. &nbsp; does not work on strings
  seeMoreDetailsSpan.textContent = '\u00A0 on altmetric attention score';
  seeMoreDetailsSpan.classList.add('sr-only');
  seeMoreDetailsLink.append(seeMoreDetailsSpan);

  var socialLinks = $('.link-to-altmetric-details-tab');
  socialLinks.each(function () {
    modifySocialLinks($(this));
  });
}

function modifySocialLinks (link) {
  var socialLinkSpan = document.createElement('span');
  socialLinkSpan.textContent = '\u00A0 - show details';
  socialLinkSpan.classList.add('sr-only');
  link.append(socialLinkSpan);
}

function modifyLogoDetails (productTitle, eachAltmetrics) {
  var altmetricsLogo = $(eachAltmetrics).find('img').eq(0);

  if (altmetricsLogo.length) {
    var altLogo = altmetricsLogo.attr('alt').split(' ');
    var altmetricScore = altLogo[altLogo.length - 1];
    altmetricsLogo.attr('alt', '');

    var logoSpan = `<span class="sr-only">Altmetric score ${altmetricScore}; more details for ${productTitle} in a new tab</span>`
    altmetricsLogo.after(logoSpan);
  }
}
