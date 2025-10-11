//Foundation JavaScript
//Documentation can be found at: http://foundation.zurb.com/docs


$(document).foundation({
  reveal: {
    multiple_opened: true // jshint ignore:line
  }
});

// Fix for IE 8/9/10, where window.location.origin is not supported.
if (!window.location.origin) {
  window.location.origin = window.location.protocol + '//' + window.location.hostname + (window.location.port ? ':' + window.location.port : '');
}
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/startsWith
// Shim for browsers that don't support the startsWith() String method (IE 11)
if (!String.prototype.startsWith) {
  String.prototype.startsWith = function (searchString, position) {
    position = position || 0;
    return this.substr(position, searchString.length) === searchString;
  };
}

var AOP = AOP || {};

AOP.closeModal = function () {
  $('.reveal-modal').foundation('reveal', 'close');
};

AOP.initAltMetric = function () {
  var $badges = $('.altmetric-badge');
  var altmetricToolTipText = AOP.translate('View Altmetric attention score details');
  $badges.not('.processed').each(function (i) {
    var $altEl = $(this);
    var doi = $altEl.data('doi');
    var isbn = $altEl.data('isbn');
    var badgeSize = $altEl.data('badge-size') || 'small'; //small, medium, large
    var container = $altEl.data('altmetric-badge-container');

    if (container) {
      $(container).hide();
    }

    if (doi || isbn) {
      $.ajax({
        type: 'GET',
        dataType: 'json',
        url: AOP.baseUrl + '/services/aop-cambridge-core/altmetric',
        data: {
          doi: doi,
          isbn: isbn
        },
        success: function (resp, status, xhr) {
          if (xhr.status !== 200 || !resp) {
            $altEl.remove();
            return false;
          }

          var imageUrl = resp.images[badgeSize];
          var detailsUrl = resp.detailsUrl;
          var score = resp.score;

          if (imageUrl && detailsUrl && score) {
            var altLink = $('<a>')
              .attr('data-ccp-qtip', '{ test: "this"}')
              .attr('title', altmetricToolTipText)
              .attr('href', detailsUrl)
              .attr('target', '_blank');

            var altDonut = $('<img/>', {
              'src': imageUrl,
              'alt': 'Altmetric attention score: ' + score
            });
            altLink.append(altDonut);
            $altEl.prepend(altLink);
          }

          if (container) {
            $(container).show();
          }
          $altEl.addClass('processed');

          if (($badges.length - 1) === i) {
            AOP.loadQtip(); // reload qtip once images injected
          }
        },
        error: function (err) {
          console.log(err);
        }
      });
    } else {
      $altEl.remove();
    }
  });

};

/*
 * Initialise Altmetric badges automatically created via the Altmetric API, which is any tag with the class
 * "altmetric-embed"
 * $container - {jQuery} - only apply within this area
 * Ref: https://api.altmetric.com/embeds.html
*/
AOP.initEmbeddedAltmetric = function ($container) {
  if (typeof _altmetric_embed_init !== 'undefined') {
    _altmetric_embed_init($container || null);
    AOP.handleAltmetricsAccessibility();
  }
};

// Add all tips to a function so it can be called when elements are added to the dom
// @note
AOP.loadQtip = function () {

  //
  // @attr          data-ccp-qtip
  // @description   an attribute based qtip for declarative configuration
  $('[data-ccp-qtip]').each(function (i, element) {

    var $element = $(element);
    var data = $element.data('ccq-qtip') || {};
    var style = $element.data('ccq-qtip-style');
    var content = $element.data('ccq-qtip-content');
    var position = $element.data('ccq-qtip-position');

    var defaults = {
      id: 'ccp-qtip_' + i,
      content: {
        attr: 'title'
      },
      position: {
        my: 'center right',
        at: 'center left'
      }
    };

    $element.qtip(defaults);


  });

  $('.download-types li a, .social li a, li.open-practice-badge-wrapper a').each(function (i, e) {
    var $e = $(e);

    if (!$e.data('loadedQtip')) {
      $e.data('loadedQtip', true);

      $e.qtip({ // create a default?
        // Force the tooltip render, so these are immediately available on the page.
        // Then we can disable the aria controls it applies, so we can implement our own accessibility on these links.
        prerender: true,
        style: {
          tip: {
            width: 20,
            height: 10
          }
        },
        content: {
          attr: 'data-hasqtip' // IE done by accessing them using getAttribute.
        },
        show: {
          event: 'mouseover focus'
        },
        hide: {
          event: 'mouseout blur'
        },
        position: {
          viewport: $(window),
          my: 'bottom center',
          at: 'top center',
          target: 'event' // my target
        }
      });
    }
  });

  $('.content-switch li a').qtip({
    // Force the tooltop render, so these are immediately available on the page
    // Core reader wants to disable these, so it can implement it's own accessibility on these links.
    prerender: true,
    style: {
      tip: {
        width: 20,
        height: 10
      }
    },
    content: {
      attr: 'data-hasqtip' //IE done by accessing them using getAttribute.
    },
    show: {
      event: 'mouseover focus'
    },
    hide: {
      event: 'mouseout blur'
    },
    position: {
      my: 'center left',
      at: 'center right',
      target: 'event' //my target
    }
  });

  $('.hasTooltipCustom').each(function () { //Notice the .each() loop, discussed below
    $(this).qtip({
      style: {
        tip: {
          width: 20,
          height: 10
        },
        classes: 'custom-tooltip'
      },
      content: {
        //text: $(this).next('.custom-tooltip')
        text: function (api) {
          var content = $(this).find('.custom-tooltip');
          return content.length !== 0 ? content.html() : content.html();
        }
      },
      show: {
        event: 'mouseover focus'
      },
      position: {
        viewport: $(window),
        adjust: { mouse: true, method: 'flip' },
        my: 'top center',
        at: 'bottom center',
        target: 'event'
      },

      hide: {
        event: 'mouseout blur',
        fixed: true,
        delay: 300
      }
    });
  });

  $('.access > li > a.hasTooltipCustom--bottom').each(function () {
    $(this).qtip({
      style: {
        tip: {
          width: 20,
          height: 10
        },
        classes: 'custom-tooltip'
      },
      content: {
        //text: $(this).next('.custom-tooltip')
        text: function (api) {
          var content = $(this).find('.custom-tooltip');
          return content.length !== 0 ? content.html() : content.html();
        }
      },
      show: {
        event: 'focus mouseover'
      },
      position: {
        viewport: $(window),
        adjust: { mouse: true, method: 'flip' },
        my: 'top center',
        at: 'bottom center',
        target: 'event'
      },
      hide: {
        event: 'blur mouseout',
        fixed: true,
        delay: 300
      }
    });
  });

  $('a.hasTooltipCustom-top').each(function () { //Notice the .each() loop, discussed below
    // Check for event type override
    var eventType = $(this).data('qtipEvent');
    // Set default qTip opts
    var qTipOpts = {
      style: {
        tip: {
          width: 20,
          height: 10
        },
        classes: 'custom-tooltip--buttons'
      },
      content: {
        text: function (api) {
          var content = $(this).next('.custom-tooltip');
          return content.length ? AOP.toolTipCloseLinkHtml + content.html() : '';
        }
      },
      position: {
        viewport: $(window),
        adjust: { mouse: true, method: 'flip' },
        my: 'bottom center',
        at: 'top center',
        target: 'event'
      },
      hide: {
        fixed: true,
        delay: 300,
        event: 'unfocus'
      },
      show: {
        event: 'click'
      },
      events: {
        show: function (event, api) {
          AOP.attatchCloseLinkToQtip(api);
          AOP.closeQtipOnEsc(api);
        }
      }
    };
    // If found, override event type for displaying qTip.
    if (eventType) {
      qTipOpts.show = {
        event: eventType
      };
    }
    $(this).qtip(qTipOpts).on('focus', function (e) {
      AOP.enableKeyboardAccessInQtipTooltip();
    });

    // Create qTip
    $(this).qtip(qTipOpts).on('click', function (e) {
      AOP.enableKeyboardAccessInQtipTooltip();
      // Disable click to stop longer page from scrolling up
      return false;
    });

    // focus on qTip when selected via keyboard
    $(this).qtip(qTipOpts).on('keydown', function (e) {
      var keyPressed = e.keyCode || e.which;
      var enterKey = 13;
      if (keyPressed === enterKey) {
        AOP.enableKeyboardAccessInQtipTooltip();
      }
    });
  });

  $('a.hasTooltipCustom-top-format').each(function () { //Notice the .each() loop, discussed below
    $(this).qtip({
      style: {
        tip: {
          width: 20,
          height: 10
        },
        classes: 'custom-tooltip--formats'
      },
      content: {
        // text: $(this).next('.custom-tooltip')
        text: function (api) {
          var content = $(this).next('.custom-tooltip');
          return content.length !== 0 ? content.html() : content.html();
        }
      },
      position: {
        viewport: $(window),
        adjust: { mouse: true, method: 'flip' },
        my: 'bottom center',
        at: 'top center',
        target: 'event'
      },
      show: {
        event: 'focus mouseover'
      },
      hide: {
        event: 'blur mouseout',
        fixed: true,
        delay: 300
      }
    });
  });

  $('.hasTooltipCustom-top-price').each(function () { //Notice the .each() loop, discussed below
    $(this).qtip({
      style: {
        tip: {
          width: 20,
          height: 10
        },
        classes: 'custom-tooltip--formats'
      },
      content: {
        //text: $(this).next('.custom-tooltip')
        text: function (api) {
          var content = $(this).parent().parent().find('.custom-tooltip');
          return content.length !== 0 ? content.html() : content.html();
        }
      },
      position: {
        viewport: $(window),
        adjust: { mouse: true, method: 'flip' },
        my: 'bottom center',
        at: 'top center',
        target: 'event'
      },
      show: {
        event: 'focus mouseover'
      },
      hide: {
        event: 'blur mouseout',
        fixed: true,
        delay: 300
      }
    });
  });

  $('.dashboard-blocks .info').each(function () { //Notice the .each() loop, discussed below
    $(this).qtip({
      style: {
        tip: {
          width: 20,
          height: 10
        }
      },
      content: {
        attr: 'data-hasqtip' //IE done by accessing them using getAttribute.
      },
      position: {
        my: 'center left',
        at: 'center right',
        target: 'event' //my target
      }
    });
  });

  $('.hasTooltipCustom-mathjax-listing').each(function () { //Notice the .each() loop, discussed below
    $(this).qtip({
      style: {
        tip: {
          width: 20,
          height: 10
        },
        classes: 'custom-tooltip--mathjax-listing'
      },
      content: {
        text: function (api) {
          var content = $(this).next('.custom-tooltip');
          return content.length !== 0 ? content.html() : content.html();
        }
      },
      position: {
        viewport: $(window),
        adjust: { mouse: true, method: 'flip' },
        my: 'bottom center',
        at: 'top center',
        target: 'event'
      },
      show: {
        event: 'click focus'
      },
      hide: {
        fixed: true,
        event: 'click blur'
      },
    }).bind('click', function (e) {
      return false;
    }); //Prevent the click
  });

  // icons with aria-labelledby should remove aria-describedby to prevent multiple announcements for screen reader users
  $('.icon[data-use-sr-only]').each(function () {
    $(this).removeAttr('aria-describedby');
  });
};

AOP.stringTrim = function (untrimmed) {
  if (untrimmed) {
    return untrimmed.toString().replace(/^\s+|\s+$/gm, '');
  }
  return '';
};

/* style checkboxes and radio inputs */
AOP.styleInputElements = function (container) {
  var selector;

  if (container) {
    selector = $('input[type=checkbox], input[type=radio]', container).not('.styled,.no-style');
  } else {
    selector = $('input[type=checkbox], input[type=radio]').not('.styled,.no-style');
  }

  selector.each(function () {
    var $input = $(this);
    
    if ($input.hasClass('styled')) {
      return;
    }
    
    $input.addClass('styled');
    
    var $nextElement = $input.next();
    var spanExists = $nextElement.length > 0 && 
                    $nextElement.is('span') && 
                    $nextElement.html() === '';
    
    if (!spanExists) {
      $input.after('<span></span>');
    }
  });
};

/* Convert a string to sentence case */
/* Eg, 'hi there' to 'Hi there' */
AOP.sentenceCase = function (sentence) {
  if (typeof sentence !== 'string') {
    return false;
  }
  sentence = sentence.toLowerCase();
  return sentence.substr(0, 1).toUpperCase() + sentence.substr(1);
};

AOP.accordionTextSwitch = function (container) {
  var $accordionTextSwitch;

  if (container) {
    $accordionTextSwitch = $(container).find('.accordion-navigation > a:not(.init)');
  } else {
    $accordionTextSwitch = $('body').find('.accordion-navigation > a:not(.init)');
  }

  $accordionTextSwitch.on('click', function (e) {
    var $clickedElement = $(e.target);
    var $clickedLink = $(this);

    // Check if the clicked element is part of a navigation link
    var $parentLink = $clickedElement.closest('a');
    var isNavigationLink = $parentLink.length > 0 && 
                          $parentLink.attr('href') && 
                          $parentLink.attr('href') !== '#' &&
                          $parentLink.attr('href').indexOf('/core') !== -1;

    // Check if this is an accordion toggle link (href starts with #)
    var isAccordionToggle = $clickedLink.attr('href') && 
                           $clickedLink.attr('href').charAt(0) === '#';
    
    // Also check if it's content inside an accordion panel (but not toggle controls)
    var isContentClick = $clickedElement.closest('.content').length > 0 && 
                        !$clickedElement.closest('.toggle-cord').length > 0;

    if (isNavigationLink) {
      return true;
    }
    
    // If it's content inside an accordion panel and NOT an accordion toggle, allow interaction
    if (isContentClick && !isAccordionToggle) {
      return true;
    }
    
    // If this is not an accordion toggle link, don't handle it here
    if (!isAccordionToggle) {
      return true;
    }
    
    e.preventDefault();
    
    var $accordionItem = $clickedLink.parent('.accordion-navigation');
    var $content = $clickedLink.siblings('.content');
    var $toggleIcon = $clickedLink.find('.toggle-cord');
    
    // Custom accordion behavior - allow multiple accordions to be open
    var isCurrentlyActive = $content.hasClass('active') || $accordionItem.hasClass('active');
    var currentExpanded = $clickedLink.attr('aria-expanded') === 'true';
    
    // Toggle only the clicked accordion
    if (isCurrentlyActive || currentExpanded) {
      // Close current accordion
      $content.removeClass('active');
      $accordionItem.removeClass('active');
      $clickedLink.attr('aria-expanded', 'false');
      $toggleIcon.removeClass('close').addClass('open-drawer');
    } else {
      // Open current accordion
      $content.addClass('active');
      $accordionItem.addClass('active');
      $clickedLink.attr('aria-expanded', 'true');
      $toggleIcon.removeClass('open-drawer').addClass('close');
      
      // Handle MathJax content
      if ($content.hasClass('do-not-mathjax')) {
        $content.removeClass('active');
        $accordionItem.removeClass('active');
      }
    }

    handleViewHideAttributes($clickedLink);
  }).addClass('init');
};

function handleViewHideAttributes(element) {
  var $textElement = $(element).find('.toggle-text');
  var ariaLabel = ($(element).attr('aria-label') || '').trim();
  var elementData = $textElement.data();
  var isViewAllToggle = elementData && elementData.viewAll && elementData.viewAll === true;
  var text = $(element).find('.toggle-text').text().trim();
  var toggledText = text === AOP.translate('View') ? AOP.translate('Hide') : AOP.translate('View');
  if (isViewAllToggle) {
    toggledText = text === AOP.translate('View all') ? AOP.translate('Hide all') : AOP.translate('View all');
  }
  $textElement.text(toggledText);
  if (ariaLabel && text) {
    var ariaLabelText = ariaLabel.replace(text, toggledText);
    $(element).attr('aria-label', ariaLabelText);
  }
}

function toggleViewHide(viewHideButton, abstractId) {
  var isViewHideButtonExpanded = viewHideButton.getAttribute('aria-expanded');
  var abstractSection = document.getElementById(abstractId);
  var accordionNavigation = viewHideButton.parentElement;

  if (isViewHideButtonExpanded === 'true') {
    viewHideButton.setAttribute('aria-expanded', 'false');
    accordionNavigation.classList.remove('active');
    abstractSection.classList.remove('active');
  } else {
    viewHideButton.setAttribute('aria-expanded', 'true');
    accordionNavigation.classList.add('active');
    abstractSection.classList.add('active');
  }

  handleViewHideAttributes(viewHideButton);
}

// Clear requested saved searches/bookmarks in the session
AOP.clearForLogin = function () {
  $.post(AOP.baseUrl + '/services/aop-cambridge-core/session/clear-for-login');
};

AOP.isReaderPage = function () {
  var re = new RegExp('core-reader');
  return re.test(window.location.pathname);
};

function showOrcidError(ErrEl) {
  ErrEl.show();
  ErrEl.parent().parent().addClass('error');
}

AOP.validateOrcidId = function (orcidId, customErrorElSelector, callback) {
  if (!orcidId) {
    return callback(true);
  }
  var customErrorEl = $(customErrorElSelector);
  var output = false;
  $.ajax({
    url: 'https://pub.orcid.org/v2.1/' + orcidId + '/person',
    headers: { 'Accept': 'application/json' },
    async: true
  }).success(function (data) {
    var results = data && data.path;
    if (results) {
      customErrorEl.hide();
      customErrorEl.parent().parent().removeClass('error');
      output = true;
      return callback(output);
    }
    showOrcidError(customErrorEl);
    return callback(output);

  }).error(function () {
    showOrcidError(customErrorEl);
    return callback(false);
  });
};

(function ($) {

  AOP.styleInputElements();
  AOP.accordionTextSwitch();
  AOP.initAltMetric();

  /* handlebars-form-helpers doesn't print out value='' for empty values */
  function putEmptyValuesInSelects() {
    if ($('select')) {
      $('select option').each(function () {
        if (!$(this).attr('value')) {
          $(this).attr('value', '');
        }
      });
    }
  }

  function formatStringInTitleCase(str) {
    if (typeof str !== 'string') {
      throw new TypeError('formatStringInTitleCase() :: argument:str :: expected:string :: actual:' + typeof str);
    }
    return str.charAt(0).toUpperCase() + str.substr(1).toLowerCase();
  }

  AOP.escapeHTML = function (str) {
    if (typeof str !== 'string') {
      throw new TypeError('escapeHTML() :: argument:str :: expected:string :: actual:' + typeof str);
    }
    return $('<div/>').html(str).text();
  }

  AOP.getOuterHTML = function ($target) {
    return $target[0].outerHTML || $('<div>').append($target.clone()).html();
  }

  $(document).on('DOMNodeInserted', function (e) {
    var hasInputs = !!$('input[type=checkbox], input[type=radio]', e.target).length;
    if (hasInputs) {
      AOP.styleInputElements(e.target);
    }
  });

  $(document).ready(function () {
    putEmptyValuesInSelects();

    $('.closeModal').on('click', function () {
      AOP.closeModal();
    });

    $(document).on('click', '.icon.share', function (e) {
      e.preventDefault();
      var $toolbox = $('.social-share-options');
      $toolbox.toggle();
      var isExpanded = $toolbox.is(':visible');
      var $shareIcon = $('.icon.share')
      $shareIcon.attr('aria-expanded', isExpanded);
    });

    $secondaryMenuIcons.on('click', function (e) {
      $(this).closest('#menu-icon').next('.dropdown-menu-mobile').find('a:first').focus();
    });

    $('.cms-block[data-block-key="aop-block-html"] a[target="_blank"]').each(function () {
      var $link = $(this);
      var isSrOnlyNotPresent = $link.find('.sr-only').length === 0;
      if (isSrOnlyNotPresent) {
        $link.append('<span class="sr-only"> (opens in a new tab)</span>');
      }
    });
  });

  AOP.loadQtip();

  var $body = $('body'),
    $window = $('window'),
    $wrapper = $('.wrapper'),
    $mainContent = $wrapper.find($('.wrapper > .main-column')),
    $sideNav = $wrapper.find($('.wrapper > .narrow-column')),
    $pageOperativesDesktop = $wrapper.find('> .page-operatives'),
    //$pageOperativesMobile = $wrapper.find('> .panel > .page-operatives'),
    $actions = $body.find($('.actions')),
    $tabsContent = $('.tabs-content'),
    $footer = $body.find($('footer')),
    $secondaryMenu = $body.find($('.secondary')),
    $subHeader = $body.find($('.sub-header')),
    $pageTabs = $body.find($('.page-tabs')),
    $corePageTabs = $body.find($('.core-page-tabs')),
    $bannerOverspill = $body.find($('.banner-overspill')),
    $optionSwitch = $wrapper.find($('.icon.switch')),
    $sideNavToggleDesktop = $wrapper.find('> .page-operatives .icon.toggle'),
    $sideNavToggleMobile = $wrapper.find('> .panel .page-operatives .icon.toggle'),
    $openMediumIcon = $tabsContent.find($('.icon.medium.open-drawer')),
    $anchorTabletIcon = $bannerOverspill.find($('a[href="#otherActions"]')),
    $openTabletIcon = $bannerOverspill.find($('.icon.tablet.open-drawer')),
    $secondaryMenuIcons = $body.find($('.menu-icon')),
    $searchExpand = $secondaryMenu.find($('.search-small')),
    $accordionExpanderFooter = $footer.find($('.accordion-expander')),
    $actionShowAllExplore = $actions.find($('.explore-all')),
    $addKeyword = $actions.find($('#addKeyword')),
    $pageTabsMobileExpand = $pageTabs.find($('.blue-square')),
    $corePageTabsMobileExpand = $corePageTabs.find($('.blue-square')),
    $mobileTab = $corePageTabs.find($('.tabs')),
    $arrowIcon = $corePageTabsMobileExpand.find('a.icon'),
    $currentMobileTabName = $('.current-mobile > a'),
    $searchForms = $body.find($('.search-form'));

  $('.carousel-facts .spinner').on('init afterChange', function (event, slick, currentSlide) {
    if (isNaN(parseFloat(currentSlide))) {
      $('.carousel-facts').find('.current').text((1) + ' of ' + slick.slideCount);
    } else {
      $('.carousel-facts').find('.current').text((currentSlide + 1) + ' of ' + slick.slideCount);
    }
  });

  $('.carousel-facts .spinner').slick({
    infinite: true,
    arrows: true
  });

  //Initialize autocomplete with local lookup:
  $('.autocomplete-suggest').each(function () {
    var autocompleteForm = $(this).closest('form');
    var moreMatchesText = 'More matches available, keep typing to refine';
    var $autocompleteOptions;
    var bookCount = 0;
    var journalCount = 0;
    var isMoreResultHint = false;
    var productTypeFilter = $(this).data('productTypes');
    $(this).devbridgeAutocomplete({
      lookup: function (query, callback) {
        var payload = {
          input: query
        };

        if (productTypeFilter) {
          payload.productType = productTypeFilter;
        }
        payload.overrideBookDateRange = true;

        $.get(AOP.baseUrl + '/services/aop-cambridge-core/search/suggest', payload, function (res) {
          var suggestions = {
            suggestions: []
          };

          if (!res) {
            return callback(suggestions);
          }

          //This lookup allows us to easily force the ordering of the autocomplete groups.
          var categories = [
            'JOURNAL',
            'BOOK'
          ];
          bookCount = 0;
          journalCount = 0;
          isMoreResultHint = false;
          if (productTypeFilter) {
            for (var i in categories) {
              var category = categories[i];

              var categoryLabelForSelectGroup = formatStringInTitleCase(category) + ' title matches';

              if (!res.hasOwnProperty(category)) {
                continue;
              }
              var numInCat = 0;
              for (var suggestion in res[category]) {
                numInCat++;
                if (numInCat > 5 && category === 'BOOK') {
                  isMoreResultHint = true
                  break;
                } else if (numInCat <= 5) {
                  var data = res[category][suggestion] || {};
                  var title = res[category][suggestion].value;
                  if (category === 'BOOK') {
                    bookCount++;
                    var biblio = [];
                    if (typeof data['bookSubtitle'] !== 'undefined' && AOP.stringTrim(data.bookSubtitle).length) { // jshint ignore:line
                      biblio.push(data.bookSubtitle);
                    }
                    if (typeof data['bookVolumeNumber'] !== 'undefined' && data.bookVolumeNumber.length && data.bookVolumeNumber.toString() !== '1') {// jshint ignore:line
                      biblio.push('Vol ' + data.bookVolumeNumber);
                    }
                    if (typeof data['bookEditionNumber'] !== 'undefined' && data.bookEditionNumber.length && data.bookEditionNumber.toString() !== '1') { // jshint ignore:line
                      biblio.push('Edition ' + data.bookEditionNumber);
                    }
                    if (biblio.length) {
                      title += ': ' + biblio.join(', ');
                    }
                  } else if (category === 'JOURNAL') {
                    journalCount++;
                  }
                  suggestions.suggestions.push({
                    value: title.toString(),
                    data: {
                      data: res[category][suggestion].data,
                      text: res[category][suggestion].value,
                      category: categoryLabelForSelectGroup
                    }
                  });
                }
              }
            }
          }
          var isMoreResultHintText = isMoreResultHint ? `<span>More matches available, keep typing to refine</span>` : ''

          var resultText = `<span>There are ${journalCount} journal title matches and ${bookCount} book title matches. Press down arrow to review results or enter to access full search results. ${isMoreResultHintText}</span>`
          $('.announce-result').html(resultText)
          callback(suggestions);
        });
      },
      beforeRender: function (container) {
        $(this).attr('aria-expanded', 'true');
        $autocompleteOptions = $(document).find('.autocomplete-suggestions');

        $($autocompleteOptions).attr('role', 'listbox');
        $($autocompleteOptions).attr('id', 'search-autocomplete');

        $(this).on('keyup', function () {
          var $autocompleteSuggestion = $(document).find('.autocomplete-selected');
          $(this).attr('aria-activedescendant', $autocompleteSuggestion.children("div").attr("id"));
        })

        // Adds the more matches available text - this prevents the element from being chosen from the suggestions
        if (isMoreResultHint) {
          $(container)
            .append(
              `<div class="autocomplete-group">
              <div class="autocomplete-more-results">
                <em>More matches available, keep trying to refine</em>
              </div>
            </div>`
            );
        }

        return container;
      },
      appendTo: '.suggestions',
      minChars: 3,
      maxHeight: 600,
      open: function () {
        var autocompleteOptionsWidth = $autocompleteOptions.width();
        $autocompleteOptions.width(autocompleteOptionsWidth + 4);
      },
      //turned off at the request of POs
      showNoSuggestionNotice: false,
      noSuggestionNotice: 'Sorry, no matching titles',
      groupBy: 'category',
      onHide: function () {
        $(this).attr('aria-expanded', 'false');
        $(this).attr('aria-activedescendant', '');
        $('.announce-result').html('');
        $(this).off('keyup');
      },
      triggerSelectOnValidInput: false,
      onSelect: function (suggestion) {
        var eventPayload = $(autocompleteForm).data('autocompleteCupEventPayload');
        var eventPayloadForAutoComplete = $.extend({}, eventPayload, {
          searchString: suggestion.value,
          usedIdentities: [] //required field but not populated for this event
        });

        $.cupEvent.dispatchEvent(eventPayloadForAutoComplete, function (err) {
          if (suggestion.data.data) {
            var eventPayloadForProduct = $.extend({}, eventPayload, {
              searchString: suggestion.value,
              usedIdentities: [],
              eventCode: 'SE-RC',
              productId: suggestion.data.data
            });
            $.cupEvent.dispatchEvent(eventPayloadForProduct, function (err) {
              window.location.href = AOP.baseUrl + '/product/' + suggestion.data.data;
            });
          } else {
            if (suggestion.value === moreMatchesText && suggestion.override) {
              $(this).val(suggestion.override);
            }
          }
        });
      },
      //Killed the normal formatting as the title needed to be implemented and we needed to strip the html from the value.
      formatResult: function (suggestion, currentValue) {
        var suggestionTitle = AOP.escapeHTML(suggestion.value);
        var suggestionHtml = `<span class="sr-only">${suggestion.data.category.split(' ')[0]} title match: ${suggestionTitle}</span> ${suggestionTitle}`
        var template = $('<div/>')
          .attr('title', suggestionTitle)
          .attr('aria-label', `${suggestion.data.category.split(' ')[0]} title match:`)
          .attr('id', suggestion.data.data)
          .attr('role', 'option')
          .html(suggestionHtml);

        return AOP.getOuterHTML(template);
      }
    });
  });

  $('img').error(function () {
    $(this).css('border', 'none');
  });

  function toggleAriaDescribedBy($toggleElement, newLabel) {
    if ($toggleElement.hasClass('refine-search')) {
      $toggleElement.attr('aria-label', newLabel);
    }
  }

  function toggleAriaLabel($toggleElement, newLabel) {
    if (!$toggleElement.hasClass('sidebar-toggle')) return;

    var currentLabel = $toggleElement.attr('aria-label');
    var toReplace = (newLabel === 'Show' ? 'Hide' : 'Show');
    $toggleElement.attr('aria-label', currentLabel.replace(toReplace, newLabel));
  }

  var excludedActions = '.book-page, .article, .chapter-page';
  //Desktop only
  //Main open close toggle side drawer
  $sideNavToggleDesktop.on('click', function (e) {
    e.preventDefault();
    var animationTime = 250;
    //Desktop view
    $(this).toggleClass('open-drawer close');
    //Open the drawer
    if ($(this).hasClass('close')) {
      $sideNav.add($mainContent).add($pageOperativesDesktop).removeClass('closed-view');
      //Actions may have been hidden in mobile view, make sure they are displayed here.
      $actions.not(excludedActions).show();
      $sideNav.hide().delay(animationTime).fadeIn(animationTime);
      //Set the hidden mobile button to have the right icon state (close)
      $sideNavToggleMobile.addClass('close');
      //Close the drawer

      // toggle aria-label for refine-search
      toggleAriaDescribedBy($(this), 'Hide refine search options');

      // toggle aria-label for sidebar-toggle
      toggleAriaLabel($(this), 'Hide');
    } else {
      $actions.not(excludedActions).fadeOut(animationTime).promise().done(function () {
        $sideNav.add($mainContent).add($pageOperativesDesktop).addClass('closed-view');
        //Set the hidden mobile button to have the right icon state (open)
        $sideNavToggleMobile.removeClass('close');
      });

      // toggle aria-label for refine-search
      toggleAriaDescribedBy($(this), 'Show refine search options');

      // toggle aria-label for sidebar-toggle
      toggleAriaLabel($(this), 'Show');
    }
  });

  //Mobile / tablet view
  //Main open close toggle side drawer
  $sideNavToggleMobile.on('click', function (e) {
    e.preventDefault();
    // $(this).toggleClass('open-drawer close');
    $(this).toggleClass('close');
    //Show the actions
    if ($(this).hasClass('close')) {
      $sideNav.add($mainContent).removeClass('closed-view');
      $actions.not(excludedActions).show();
      //Set the hidden desktop button to have the right icon state (close)
      $sideNavToggleDesktop.removeClass('open-drawer').addClass('close');
      //Hide the actions

      // toggle aria-label for refine-search
      toggleAriaDescribedBy($(this), 'Hide refine search options');

      // toggle aria-label for sidebar-toggle
      toggleAriaLabel($(this), 'Hide');
    } else {
      $actions.not(excludedActions).hide().promise().done(function () {
        $sideNav.add($mainContent).addClass('closed-view');
        //Set the hidden desktop button to have the right icon state (close)
        $sideNavToggleDesktop.removeClass('close').addClass('open-drawer');
      });
      // toggle aria-label for refine-search
      toggleAriaDescribedBy($(this), 'Show refine search options');

      // toggle aria-label for sidebar-toggle
      toggleAriaLabel($(this), 'Show');
    }
  });

  // Only allow search form to be submitted if user has entered at least one character
  $subHeader.find('input[type=submit]').on('click', function (e) {
    if ($subHeader.find('input[name=q]').val().length === 0) {
      e.preventDefault();
    }
  });

  // Only allow search form to be submitted if user has entered at least one character
  $($searchForms).each(function () {
    var self = this;
    if ($(this).hasClass('prevent-empty-search')) {
      $(this).find('input[type=submit]').on('click', function (e) {
        if ($.trim($(self).find('input[name=q]').val()).length === 0) {
          e.preventDefault();
        }
      });
    }
    //log event
    $(this).submit(function (e, evented) {
      var self = this;
      var eventPayload = $(this).data('cupEventPayload');

      //If there's no payload, don't bother.
      if (eventPayload) {
        $.extend(eventPayload, {
          searchString: $(self).find('input[name="q"]').val(),
          usedIdentities: [] //required field but not populated for this event
        });

        if (!evented) {
          e.preventDefault();
          $.cupEvent.dispatchEvent(eventPayload, function () {
            $(self).trigger('submit', true);
          });
        }
      }
    });
  });

  const waitForElementProp = (selector, prop) => {
    return new Promise(resolve => {
        const element = document.querySelector(selector)
        if (element && element[prop]) {
            resolve(element[prop]);
            return;
        }

        const observer = new MutationObserver(mutations => {
          const oberverElement = document.querySelector(selector)
            if (oberverElement && oberverElement[prop]) {
                observer.disconnect();
                resolve(oberverElement);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
}

  //Mobile menu secondary
  const closeMobileMenuEventListener = (e) => {
    e.preventDefault();
    if($('.menu-icon').attr('aria-expanded') === 'true') {
      $('.menu-icon').trigger('click');
    }
  }
  waitForElementProp('global-header', 'shadowRoot').then(elem => {
    elem.querySelectorAll('.gh-accordionHeader').forEach(header => {
      header.addEventListener('click', closeMobileMenuEventListener)
    })
    // close global header on close menu
    $('.menu-icon').on('click', e => {
      if($('.menu-icon').attr('aria-expanded') === 'true') {
        elem.querySelectorAll('.gh-accordionHeader button').forEach(button => {
          if($(button).attr('aria-expanded') === 'true') {
            $(button).trigger('click');
          }
        })
      }
    })
  })

  $secondaryMenuIcons.each(function (i, e) {
    $(this).on('click', function (e) {
      if(!e.originalEvent && e.target.className !== 'menu-icon open active') {
        return
      }
      var $anyDropdown = $body.find('.dropdown-menu-mobile'),
        $nearestDropdown = $(this).parent().parent().next(),
        $backdrop = '<div class="backdrop"></div>';
      $(this).attr('aria-expanded', 'true');
      $(this).toggleClass('open');

      if (!$anyDropdown.hasClass('open-drawer')) {
        document.body.classList.add('overflow-hidden');
        $nearestDropdown.toggleClass('open-drawer').parent().append($backdrop);

        if ($(this).hasClass('identities')) {
          $nearestDropdown.removeClass('personal').addClass('identities');
        }
        if ($(this).hasClass('personal')) {
          $nearestDropdown.removeClass('identities active').addClass('personal');
        }

        if (!$(this).hasClass('identities')) {
          $(this).attr('aria-label', 'close site navigation');
          $('.search-small').attr('tabindex', '-1');
          AOP.focusTrap("dropdown-menu-mobile", $('.menu-icon')[0])
          $(this).addClass('active');
          const mobileSearch = document.querySelector('.mobile-search')
          if(mobileSearch) {
            mobileSearch.addEventListener('click', closeMobileMenuEventListener)
          }
        }

      } else {
        document.body.classList.remove('overflow-hidden');
        $anyDropdown.removeClass('open-drawer').parent().find('.backdrop').remove();
        $(this).removeClass('active');
        $(this).attr('aria-expanded', 'false');
        $(this).attr('aria-label', 'open site navigation');
        $('.search-small').attr('tabindex', '0');
        $('.menu-icon').off('keydown')
      }
    });
  });

  $('.js-toggle-show').on('click', function () {
    //$('.accordion').find('content').removeClass('active');
    var $button = $(this);
    if ($('.accordion').data('showing') === true) {
      $('.accordion').find('.content').removeClass('active');
      $button.text('Expand full list');
      $('.accordion').data('showing', false);
    } else {
      $('.accordion').find('.content').addClass('active');
      $button.text('Hide full list');
      $('.accordion').data('showing', true);
    }
  });

  //mathJax or any other item using this switch type
  $optionSwitch.on('click', function () {
    $(this).toggleClass('off on');
  });

  $('.js-outglow').on('focus', function () {
    $(this).closest('.js-outglow-parent').toggleClass('glow');
  });

  $('.js-outglow').on('blur', function () {
    $(this).closest('.js-outglow-parent').toggleClass('glow');
  });

  //Footer Mobile accordion
  $openMediumIcon.on('click', function () {
    $(this).toggleClass('open-drawer close');
  });

  //Footer Mobile accordion
  $openTabletIcon.on('click', function () {
    $(this).toggleClass('open-drawer close');
    if ($(this).hasClass('close')) {
      $anchorTabletIcon.attr('aria-label', 'Hide other actions');
    } else {
      $anchorTabletIcon.attr('aria-label', 'Show other actions');
    }
  });

  // Open/close icons for viewing other actions using mobile view
  $anchorTabletIcon.on('click', function () {
    $openTabletIcon.toggleClass('open-drawer close');
    if ($openTabletIcon.hasClass('close') || $anchorTabletIcon.attr('aria-expanded') === 'false') {
      $anchorTabletIcon.attr('aria-label', 'Hide other actions');
    } else {
      $anchorTabletIcon.attr('aria-label', 'Show other actions');
    }
  });

  //Toggle for Journal Accordians
  $('.toggle-cord').on('click', function () {
    $(this).toggleClass('open-drawer close');
    $(this).toggleClass('open-drawer close');
  });

  //Journal back-issues nested accordion
  $accordionExpanderFooter.on('click', function () {
    $(this).find('.icon').toggleClass('close open-drawer');
  });

  $('.js-expand').on('click', function (e) {
    e.preventDefault();
    // Get both desktop and responsive buttons
    var expandButtons = $('.js-expand');
    var expandButtonsText = expandButtons.find('span');
    var buttonText;
    var ariaExpandText = $(this).data('ariaExpandText');
    var ariaCollapseText = $(this).data('ariaCollapseText');
    if (!$(this).data('expanded')) {
      $('.nested-accordion .content').removeClass('active').addClass('active');
      $('.nested-accordion [aria-expanded="false"]').attr('aria-expanded', 'true');
      $('.nested-accordion .toggle-cord').removeClass('open-drawer').addClass('close');
      buttonText = AOP.translate('Collapse list');
      $(this).data('expanded', true);
      expandButtonsText.text(buttonText);
      expandButtons.attr('aria-label', ariaCollapseText);
      expandButtons.data('expanded', true);
    } else {
      $('.nested-accordion .content').removeClass('active');
      $('.nested-accordion [aria-expanded="true"]').attr('aria-expanded', 'false');
      $('.nested-accordion .toggle-cord').removeClass('close').addClass('open-drawer');
      buttonText = AOP.translate('Expand full list');
      $(this).data('expanded', false);
      expandButtonsText.text(buttonText);
      expandButtons.attr('aria-label', ariaExpandText);
      expandButtons.data('expanded', false);
    }
    return false;
  });

  //Mobile sub-header toggle
  const searchExpandEventHandler = function () {
    if ($subHeader.hasClass('slide-up')) {
      $subHeader.removeClass('slide-up').addClass('slide-down');
      $(this).addClass('active');
      $(this).children('span').addClass('active');
      $(this).attr('aria-expanded', 'true');
      $(this).attr('aria-label', 'Close site search');
    } else {
      $subHeader.removeClass('slide-down').addClass('slide-up');
      $(this).removeClass('active');
      $(this).children('span').removeClass('active');
      $(this).attr('aria-expanded', 'false');
      $(this).attr('aria-label', 'Open site search');
    }
  }
  $searchExpand.on('click', searchExpandEventHandler);
  $searchExpand.on('keydown', e => {
    if (e.code === 'Enter') {
      searchExpandEventHandler()
    }
  });

  // Journal page / mobile / fix state of arrow icon
  $('.tab-title.current-mobile a').on('click', function () {
    var mobileTabDisplay = $mobileTab.css('display') === 'none' ? 'block': 'none';
    $mobileTab.css('display', mobileTabDisplay)

    $arrowIcon.toggleClass('open');

    $(this).attr('aria-expanded', $arrowIcon.hasClass('open'));
  });

  //Mobile & Desktop Page tabs menu
  $pageTabs.on('click', '.tab-title a', function (e) {
    e.preventDefault();
    $pageTabs.find('.tabs').toggleClass('open-drawer');
    if (!$(this).hasClass('icon')) {
      $currentMobileTabName.text($(this).text());
      if ($(window).width() < 1024) {
        $pageTabsMobileExpand.find('a').toggleClass('open'); //Return the arrow to correct state.
        $('.icon.toggle.close').click(); //Hide actions that were open on previous page.
      }
    } else {
      $(this).toggleClass('open');
    }
  });

  var loaded = false;
  $actionShowAllExplore.each(function () {
    $(this).on('click', function (e) {
      var limit = 5;
      if ($(this).data('limit')) {
        limit = $(this).data('limit');
      }

      var $showAll = $(this);
      var $buttonTextAction = $(this).find('.toggle-text');
      var $buttonTextAmount = $(this).find('.excerpt-type');
      var $exploreLink = $(this).find('a[href="#explore"]');
      var $exploreLinkAriaLabel = $exploreLink.attr('aria-label') || '';
      var $buttonIcon = $(this).find('.icon');
      e.preventDefault();
      var $thisActionsList = $showAll.closest($actions).find('ul');
      if (loaded) {
        if ($thisActionsList.find('li:eq(' + (limit) + ')').is(':hidden')) {
          $thisActionsList.find('li:gt(' + (limit - 1) + ')').fadeIn(300, function () {
            $buttonTextAction.text('Show');
            $buttonTextAmount.text('fewer');
            $exploreLink.attr('aria-label', $exploreLinkAriaLabel.replace('all', 'fewer'));
            $buttonIcon.addClass('close');
          });
        } else {
          $thisActionsList.find('li:gt(' + (limit - 1) + ')').fadeOut(300, function () {
            $buttonTextAction.text('Show');
            $buttonTextAmount.text('all');
            $exploreLink.attr('aria-label', $exploreLinkAriaLabel.replace('fewer', 'all'));
            $buttonIcon.removeClass('close');
          });
        }
      } else {
        $($thisActionsList).find('li:gt(' + (limit - 1) + ')').fadeIn(300, function () {
          $buttonTextAction.text('Show');
          $buttonTextAmount.text('fewer');
          $exploreLink.attr('aria-label', $exploreLinkAriaLabel.replace('all', 'fewer'));
          $buttonIcon.addClass('close');
        });
        loaded = true;
      }
    });
  });

  $actions.on('submit', $addKeyword, function (e) {
    e.preventDefault();
    var keyword = $addKeyword.val();
    if (keyword && keyword.length) {
      var addedKeywordListItem = '<li><a href="#"><span class="icon delete"></span>' + keyword + '</a></li>';
      $('.keywords').append(addedKeywordListItem);
    }
  });

  $('.keywords').on('click', 'li a', function (e) {
    e.preventDefault();
    $(this).parent().remove();
  });

  //Really need a better fix for this...
  $('ul#text-resizer-controls').each(function () {
    $(this).children().textresizer({
      target: '.large-12 .wrapper',
      type: 'cssClass',
      sizes: [
        'standard-text',
        'large-text'
      ],
      selectedIndex: 1
    });
  });

  $('.dashboard-blocks .info').one('click mouseenter', function (e) {
    $(this).removeClass('unread');
  });

  // Blurred Cover
  $('.blurred-cover').foggy({
    blurRadius: 8, //In pixels.
    opacity: 0.8, //Falls back to a filter for IE.
    cssFilterSupport: true //Use "-webkit-filter" where available.
  });

  // Disabled this - as it already handled by AOP.accordionTextSwitch()
  //View/hide abstract toggle on book/journal and search pages
  // $('.abstract').on('click', 'a', function (e) {
  //   e.preventDefault();
  //   var toggleText = $(this).find('.toggle-text');
  //   var icon = $(this).find('>span');
  //   icon.toggleClass('close');
  //
  //   $(this).next().toggle();
  // });

  // On listings, open embedded links within abstracts/extracts, in a new window/tab.
  $('.results').on('click', '.content .abstract a', function (e) {
    e.preventDefault();
    window.open($(this).attr('href'), '_blank');
  });

  // On article pages, open embedded links within abstracts/extracts, in a new window/tab.
  $('.article-overview .abstract a').on('click', function (e) {
    e.preventDefault();
    window.open($(this).attr('href'), '_blank');
  });

  // Detect click on institution logo in dropdown
  $('#institutions-dropdown img, .show-for-small.institutions img').on({
    click: function () {
      var link = $(this).parent().parent().find('a');
      if (link.attr('data-no-link') === undefined) {
        window.open(link.attr('href'), '_blank');
      }
    },
    mouseenter: function () {
      var link = $(this).parent().parent().find('a');
      if (link.attr('data-no-link') === undefined) {
        $(this).css({ cursor: 'pointer' });
      } else {
        $(this).css({ cursor: 'default' });
      }
    }
  });

  // Peform any actions on the org logo
  var orgLogoCheck = function () {
    var deviceLogo = $('a.account-loggedin.identities');
    // Check if we are dealing with a screen that's too small to show
    // the org logo in the header.
    if (window.orientation === 0 && $(window).width() <= 320) {
      deviceLogo.addClass('no-logo');
    } else {
      deviceLogo.removeClass('no-logo');
    }
  };

  // Set the correct tab name on mobile
  var tabName;
  // Tabs can be rendered under two classes, check corePageTabs if
  // no data found in the expected place

  var tabIdx = 0;
  var $collection = $pageTabs.length !== 0 ? $pageTabs : $corePageTabs;
  $.each($collection, function () {
    tabName = $(this).find('.tab-title.active').text();
    $($currentMobileTabName.get(tabIdx)).text(tabName);
    tabIdx++;
  });

  // Get and display the first available logo/link in the institution list
  var institutions = $('.institution-name ul li.institution'),
    primaryLogo = $('.institution-primary-logo'),
    deviceLogo = $('a.account-loggedin.identities'),
    link,
    image,
    institutionName,
    imageFound = false;
  if (institutions.length > 0) {
    $.each(institutions, function () {
      link = $(this).find('a');
      image = $(this).find('img');
      institutionName = link.text();
      if (!!image.attr('src')) {
        primaryLogo.find('a').append('<img src="' + image.attr('src') + '" alt="' + institutionName + '">');
        deviceLogo.find('img').attr('src', image.attr('src')).attr('alt', institutionName);
        // Show/hide the org logo
        orgLogoCheck();
        if (!link.attr('data-no-link') && !!link.attr('href')) {
          primaryLogo.find('a').attr('href', link.attr('href')).attr('target', '_blank');
        } else {
          primaryLogo.find('a').attr('data-no-link', true);
        }
        primaryLogo.show();
        imageFound = true;
        return false;
      }
    });
  }
  if (!imageFound) {
    primaryLogo.remove();
  }

  // Listen for device orientation changes, perform any required actions.
  if (window.addEventListener) {
    window.addEventListener('orientationchange', function () {
      orgLogoCheck();
    }, false);
  }

  $('.aop-toggle-text').each(function () {
    var $event = $(this);
    var $buttonText = $event.find('.button-text');
    var textOn = $event.data('toggleOn') || $buttonText.html();
    var textOff = $event.data('toggleOff');
    var ariaEntityName = $event.data('ariaEntityName');

    // Set the ARIA label text. If available, append the ARIA entity name (eg 'journals') to the existing toggle text.
    var setAriaLabel = function (el, labelText, entityName) {
      if (typeof labelText === 'string' && el) {
        el.attr('aria-label', labelText.trim() + (entityName ? ' ' + entityName : ''));
      }
    };

    // Init ARIA label text
    setAriaLabel($event, textOn, ariaEntityName);

    if (!textOn || !textOff) {
      console.warn('Specify data attributes for toggling text', $event);
    } else {
      $event.click(function (e) {
        e.preventDefault();
        if ($buttonText.html() === textOn) {
          $buttonText.html(textOff);
          setAriaLabel($event, textOff, ariaEntityName);
        } else {
          $buttonText.html(textOn);
          setAriaLabel($event, textOn, ariaEntityName);
        }
      });
    }
  });

  $('.aop-toggle-class').each(function () {
    var $event = $(this);
    var cssOn = $event.data('toggleClassOn');
    var cssOff = $event.data('toggleClassOff');

    if (!cssOn || !cssOff) {
      console.warn('Specify data attributes for toggling css', $event);
    } else {
      $event.click(function (e) {
        e.preventDefault();
        $event.hasClass(cssOn) ? $event.removeClass(cssOn).addClass(cssOff) : $event.removeClass(cssOff).addClass(cssOn); // jshint ignore:line
      });
    }
  });

  $('.aop-toggle-show').each(function () {
    var $event = $(this);
    var $target = $($event.data('toggleTarget'));

    if ($target.length > 0) {
      $event.click(function (e) {
        e.preventDefault();
        $target.toggleClass('hide');
      });
    }
  });

  $('.aop-toggle-replace').each(function () {
    var $event = $(this);
    var targets = $event.data('replaceTarget').split(',');
    if (!targets.length) {
      return false;
    }

    var $initial = $(targets[0]);
    var $target = $(targets[1]);

    if ($target.length && $initial.length) {
      $event.click(function (e) {
        e.preventDefault();
        $target.toggleClass('hide');
        $initial.toggleClass('hide');
      });
    }
  });

  $('.aop-toggle-action').each(function () {
    var $event = $(this);
    var action = $event.data('toggleAction');
    var $target = $($event.data('toggleTarget'));
    if (!action) {
      console.warn('Specify toggle action', $event);
    } else {
      $event.click(function (e) {
        e.preventDefault();
        switch (action) {
          case 'showHide':
            $target.toggleClass('hide');
            break;
        }
      });
    }
  });

  $('body').on('click', '.check-library-catalogue', function (e) {
    e.preventDefault();
    $(this).parent().find('.check-access-openurl').slideToggle();
  });

  $('body').on('click', '.listing-citation-modal', function (e) {
    e.preventDefault();
    var $button = $(this);
    var $modal = $('#productCitations');
    var $allCitedByButtons = $('.listing-citation-modal');
    var $loader = $modal.find('.loader');
    var $modalContent = $modal.find('.citation-content');
    var serviceUrl = AOP.baseUrl + '/services/aop-cambridge-core/citations';
    var $errorMessages = $modal.find('#ajaxMessages');
    var alertOpts = {
      scroll: false,
      alertType: 'error',
      alertElement: $errorMessages
    };
    // Get button data
    var doi = $button.attr('data-doi');
    var productType = $button.attr('data-product-type');
    var productId = $button.attr('data-product-id');
    var pagePath = $button.attr('data-page-path');
    var citationsLoaded = $button.attr('data-citations-loaded');
    // Citation content already loaded for this product, just open the modal.
    if (citationsLoaded === 'yes') {
      $modal.foundation('reveal', 'open');
    } else {
      // Hit the service and get citation content
      $modalContent.html('');
      $loader.show();
      $modal.foundation('reveal', 'open');
      $errorMessages.find('>div').remove();
      $.post(serviceUrl, { doi: doi, productType: productType, productId: productId, pagePath: pagePath }).done(function (response) {
        $loader.hide();
        if (response.success) {
          $modalContent.html(response.html);
          $allCitedByButtons.attr('data-citations-loaded', 'no');
          $button.attr('data-citations-loaded', 'yes');
        } else {
          AOP.createAlertBox('There was a problem displaying the citations. Please <a href="' + AOP.baseUrl + '/contact">contact customer services</a>.', alertOpts);
        }
      });
    }
  });


  /*,
   * Bind desktop, mobile, and tablet behaviour to the secondary menu dropdowns
   * @param {string} elementId - id of dropdown link
   * @param {object} opts - dropdown behaviour
   *  top: top offset positioning for dropdown
   *  left: left offset positioning for dropdown
   */
  var initSecondaryMenuDropdowns = function (elementId, opts) {
    if (!elementId || typeof elementId !== 'string') {
      return;
    }
    // Bind the parent <li> element as the button, it's a larger area, so easier to hover/tap.
    // Define separate buttons for touch (mobile/tablet) and desktop - use modernizr classes to check this.
    var $button = $('#' + elementId).parent()


    function outsideClickHandler(e) {
      var keyPressed = e.keyCode || e.which;
      if (e.type === 'click' || e.type === 'touchend' || keyPressed === 32) {
        if ($(e.target).closest($('#' + elementId).parent()).length === 0) {
          dropdown.hide();
        }
      }
    }

    // Hide, show and position the dropdown
    var dropdown = {
      // Set dropdown area
      $el: $('#' + elementId).next(),
      // Set top/left positioning offsets
      top: opts.top || 0,
      left: opts.left || 0,
      // Test id dropdown is hidden/visible
      isHidden: function () {
        return this.$el.attr('aria-hidden') === 'true';
      },
      // Hide the dropdown
      hide: function () {
        this.$el.css('display', 'none').attr('aria-hidden', 'true')
        $('html').off('click keydown touchend', outsideClickHandler);
        this.$el[0].style.setProperty('display', 'none', 'important');
        $('#' + elementId).attr('aria-expanded', 'false');
      },
      // Show the dropdown
      show: function () {
        this.$el.css({
          left: $('#' + elementId).position().left + this.left,
          top: this.top,
          display: 'inherit'
        }).attr('aria-hidden', 'false');
        this.$el[0].style.setProperty('display', 'inherit', 'important');

        $('html').on('click keydown touchend', outsideClickHandler)
        $('#' + elementId).attr('aria-expanded', 'true');
      },
      // Toggle dropdown visibility state
      toggle: function () {
        if (this.isHidden()) {
          this.show();
          return;
        }
        this.hide();
      }
    };

    // Anything detected as desktop device, handle hovers and clicks.
    $button.bind('click keydown', function (e) {
      // This is a dirty fix for improper HTML structure which contains self-contradictory nested items (desktop vs mobile)
      // need to check if this was setup for visible desktop or hidden mobile elment
      // for mobile we prevent action so we don't display the same nested menu again
      const canProcessClick = $('#' + elementId).css('display') !== 'none';
      if (!canProcessClick) {
        return;
      }

      var keyPressed = e.keyCode || e.which;

      if (keyPressed === 27) {
        dropdown.hide();
      }

      if (e.type === 'click' || keyPressed === 32) {
        dropdown.toggle();
        // space
        if (keyPressed === 32) {
          // prevent scrolling
          e.preventDefault();
        }
      }


    });
  };


  initSecondaryMenuDropdowns('r-user-dropdown-icon-mobile-view', { top: 40, left: -12 });
  initSecondaryMenuDropdowns('r-user-dropdown-icon-desktop-view', { top: 40, left: -12 });
  initSecondaryMenuDropdowns('r-user-dropdown-icon-homepage', { top: 40, left: -12 });
  initSecondaryMenuDropdowns('r-user-dropdown-icon', { top: 40, left: -12 });
  initSecondaryMenuDropdowns('institution-current', { top: 31 });
  initSecondaryMenuDropdowns('institution-current-mobile', { top: 31 });
  initSecondaryMenuDropdowns('institution-current-desktop', { top: 31 });

  // If the user clicks cancel in a login modal, clear out the requested saved search or bookmarks.
  $(document).on('close.fndtn.reveal', '#login-modal[data-reveal]', function () {
    AOP.clearForLogin();
  });

  $(window).resize(function() {
    if ($(window).width() <= 1024 && $arrowIcon.hasClass('open')) {
      $mobileTab.css('display', "block")
    }else{
      $mobileTab.css('display', "")
    }
  });

})(jQuery);

// This is added to handle cases where the jacket is defined but the file location gives a 404
$(window).on('load', function () {
  var images = $('[data-enable-cup-replacement="true"]');
  for (var i = 0; i < images.length; i++) {
    var img = $(images)[i];
    if (!img.complete || typeof img.naturalWidth === 'undefined' || img.naturalWidth === 0) {
      $(img).attr('src', AOP.baseUrl + '/cambridge-core/public/images/jacket_blank_300x448.jpg');
      $(img).css('width', 180);
    }
  }
});
//https://www.nczonline.net/blog/2013/01/15/fixing-skip-to-content-links/
window.addEventListener('hashchange', function (event) {
  var element = document.getElementById(location.hash.substring(1));
  if (element) {
    if (!/^(?:a|select|input|button|textarea)$/i.test(element.tagName)) {
      element.tabIndex = -1;
    }
    element.focus();
  }
}, false);

// Closes the modal when click outside of it
$('body').on('click', function (e) {
  if (!$(e.target).closest('.reveal-modal').length && !$('#cms-modal').is(":visible")) {
    $('.reveal-modal').foundation('reveal', 'close')
  }
})
