var AOP = AOP || {}

AOP.citationTool = {
  // Services
  SERVICE_URL_CITE: AOP.baseUrl + '/services/aop-citation-tool/cite',
  SERVICE_URL_DOWNLOAD: AOP.baseUrl + '/services/aop-citation-tool/download',
  // Classes
  CITATION_TOOL_MODAL_ID: 'exportCitation',
  CITATION_UI_CLASS: 'citation-ui',
  PRODUCT_BUTTON_CLASS: 'export-citation-product',
  COMPONENT_BUTTON_CLASS: 'export-citation-component',
  EXPORT_ALL_BUTTON: 'export-all-citations',
  CITATION_DATA_DISPLAY_ID: 'citationDataDisplay',
  SELECT_CITATION_STYLE_CLASS: 'select-citation-style',
  CITATION_HTML_TEXT_ID: 'citationText',
  CITATION_CLIPBOARD_SR_INFO: 'copyToClipboardInfo',
  COPY_CITATION_ID: 'copyCitationData',
  COMPONENT_LIST_BUTTON_CLASS: 'export-citation-list',
  // Cookie settings
  COOKIE_NAME: 'CORE_CITATION_STYLE',
  COOKIE_EXPIRES: 365,
  // Kill switch
  SHOULD_USE_CITATION_TOOL: AOP.shouldUseCitationTool || true,

  initialised: false,
  productIds: [],
  citationStyleOptions: [],

    // We can only generate a citation for the following product types
  // Product types are being repeated as the My Core Bookmarks page uses different type keys (the lower case ones).
  validProductTypes: ['JOURNAL_ARTICLE', 'BOOK_PART', 'BOOK', 'ELEMENT', 'book', 'chapter', 'article', 'element'],
  invalidProductsSelected: false,

  $exportCitationTool: null,
  $citationUi: null,
  $loader: null,
  $selectCitationStyle: null,
  $citationData: null,
  $citationDataDisplay: null,
  $copyToClipboardMessageForSr: null,

  messages: {
    error: 'Could not export the citation. Please <a href="' + AOP.baseUrl + '/contact">contact customer services</a>.',
    noContent: 'No content selected. Please select items you would like to export a citation for.',
    copyText: 'Copying text is not supported in this browser',
    citationToolSwitchOff:
      'Apologies, this service is temporarily unavailable. We are working to restore this as soon as possible.',
    invalidProductsSelected:
      'A citation could not be generated for these products. Citations can only be generated for articles, books or chapters.'
  },

  /**
   * Create the citation tool modal and add it to the DOM
   */
  initContainer: function () {
    const _self = this
    _self.$exportCitationTool = $('#' + _self.CITATION_TOOL_MODAL_ID)

    if (_self.$exportCitationTool.length === 0) {
      const container = `
      <div
        id="${_self.CITATION_TOOL_MODAL_ID}"
        class="reveal-modal medium"
        data-reveal
        role="dialog"
        aria-labelledby="exportCitationModalHeader"
        aria-describedby="exportCitationModalDescription"
      >
        <div tabindex="-1" id="citation-modal" class="modal-outline">
          <div class="header">
            <h1 class="heading_07" id="exportCitationModalHeader">${AOP.translate('Citation Tools')}</h1>
            <p id="exportCitationModalDescription">${AOP.translate('Copy and paste a formatted citation or download in your chosen format')}</p>
          </div>
          <div class="row wrapper no-padding-top">
            <div class="loader">
              <img src="/core/system/public/img/ajax_loader_gray_256.gif" alt="">
              <span role="alert" aria-live="polite">Loading citation...</span>
            </div>
            <div class="small-12 columns content" style="display:none"></div>
          </div>
          <a href="#" class="close-reveal-modal" aria-label="Close Citation Tools"><span aria-hidden="true">×</span></a>
        </div>
      </div>`

      $('body').append(container)
      _self.$exportCitationTool = $('#' + _self.CITATION_TOOL_MODAL_ID)
      _self.$loader = _self.$exportCitationTool.find('.loader')
    }

    $('.close-reveal-modal').on('click', function () {
      $('.cite-button').focus()
      $('.' + _self.CITATION_TOOL_MODAL_ID).off('keydown')
    })
  },

  /**
   * Bind citation tool events
   */
  initEvents: async function () {
    const _self = this

    // Wrapper for all citation sections
    _self.$citationUi = $('.' + _self.CITATION_UI_CLASS)
    // The currently selected style
    _self.$selectCitationStyle = $('#selectCitationStyle')

    _self.multilingualTranslation()

    // Formatted citation results
    _self.$citationData = $('.citation-data')
    // When the formatted citation is rendered
    _self.$citationDataDisplay = $('#' + _self.CITATION_DATA_DISPLAY_ID)

    // Accessibility
    _self.$copyToClipboardMessageForSr = $('#clipboardAlertForSr')

    // Copy citation data to clipboard
    _self.$exportCitationTool.on('click', '#' + _self.COPY_CITATION_ID, function (e) {
      e.preventDefault()
      _self.clearMessages()
      const $button = this
      const successMessage = 'The citation has been copied to your clipboard'
      // Remove background formatting, so this is not copied.
      _self.$citationDataDisplay.addClass('transparent')
      _self.copyRichText(_self.CITATION_HTML_TEXT_ID, function (err) {
        if (err) {
          _self.showMessage(err)
          _self.copyToClipboardAlertForSr(err)
        } else {
          _self.showMessage(successMessage, { alertType: 'info' })
          _self.copyToClipboardAlertForSr(successMessage)
        }
        // Add back background colour
        _self.$citationDataDisplay.removeClass('transparent')
        // Send focus back to clicked button
        $($button).focus()
      })
    })

    // Export citation data in selected format
    _self.$exportCitationTool.find('.export').on('click', function (e) {
      e.preventDefault()

      const $button = $(this)
      const type = $button.data('exportType') || false
      const citationStyle = _self.getSavedCitationStyle().key

      if (!type) {
        _self.showMessage(_self.messages.error)
        return
      }

      const url = `${_self.SERVICE_URL_DOWNLOAD}/?downloadType=${type}&productIds=${_self.productIds}&citationStyle=${citationStyle}`
      window.location = url
    })

    // On load, set the last used citation style.
    const citationStyle = _self.getSavedCitationStyle()
    _self.$selectCitationStyle.val(citationStyle.key)
  },

  multilingualTranslation: function () {
    const _self = this
    const translateText = function (domObject) {
      const originalText = domObject.text()
      const translation = AOP.translate(originalText)

      return domObject.text().replace(originalText, translation)
    }
    $('#' + _self.COPY_CITATION_ID).text(function () {
      return translateText($(this))
    })
    $('#citation-export-options-heading-download').text(function () {
      return translateText($(this))
    })
  },

  /**
   * Save preferred citation style in cookie
   * @param citationStyleKey
   * @param citationStyleName
   */
  saveCookie: function (citationStyleKey, citationStyleName) {
    const _self = this
    $.cookie(
      _self.COOKIE_NAME,
      JSON.stringify({
        key: citationStyleKey,
        displayName: citationStyleName
      }),
      { expires: _self.COOKIE_EXPIRES, path: '/' }
    )
  },

  /**
   * Get the saved citation style from cookie
   * @returns {{key: string, displayName: string}}
   */
  getSavedCitationStyle: function () {
    const _self = this
    const citationStyle = $.cookie(this.COOKIE_NAME)
    const defaultCitationStyle = {
      key: _self.citationStyleOptions[0].key,
      displayName: _self.citationStyleOptions[0].displayName
    }

    // Set data if cookie found
    if (typeof citationStyle === 'string') {
      const parsedCitationStyle = JSON.parse(citationStyle)

      const citationStyleFromOptions = _self.citationStyleOptions.find(citation => citation.key === parsedCitationStyle.key)

      // Set default if cookie value in not on options list
      return citationStyleFromOptions || defaultCitationStyle
    }

    // Set default if no cookie found
    return defaultCitationStyle
  },

  /**
   * Attempt to copy the currently selected text to the clipboard
   * @param callback
   * @returns {*}
   */
  copyText: function (callback) {
    if (!document.execCommand) {
      return callback(new Error(this.messages.copyText))
    }
    if (!document.execCommand('copy')) {
      return callback(new Error(this.messages.copyText))
    }
    return callback()
  },

  /**
   * Attempt to copy the text in the target element to the clipboard, with HTML formatting.
   * @param elementId
   * @param callback
   * @returns {*}
   */
  copyRichText: function (elementId, callback) {
    const copyError = new Error(this.messages.copyText)
    if (!window.getSelection || !document.execCommand || !document.createRange) {
      return callback(copyError)
    }

    const range = document.createRange()
    // Focus on the element where we want to copy the text from
    $('#' + elementId).focus()

    try {
      // Clear selection (Chrome fails otherwise)
      window.getSelection().removeAllRanges()
      // Select the text in the element
      range.selectNode(document.getElementById(elementId))
      if (!range) {
        return callback(copyError)
      }
      try {
        window.getSelection().addRange(range)
      } catch (error) {
        return callback(copyError)
      }
      // Copy to clipboard
      if (!document.execCommand('copy')) {
        return callback(copyError)
      }
      // Clear selected text
      window.getSelection().removeAllRanges()
      return callback()
    } catch (error) {
      return callback(copyError)
    }
  },

  /**
   * Create a accessible alert for screen readers, on citation copy.
   * @param message
   */
  copyToClipboardAlertForSr: function (message) {
    // We have to clear/set the role, for the screenreader to pick this up, if it's clicked more than once.
    this.$copyToClipboardMessageForSr.attr('role', '').attr('role', 'alert').text(message)
  },

  /**
   * Simply open the citation tool modal
   */
  openModal: function () {
    this.clearMessages()
    const focusTimeOut = 1000
    this.$exportCitationTool.foundation('reveal', 'open')
    var $citationModal = $('#citation-modal')

    setTimeout(function () {
      $citationModal.focus()
    }, focusTimeOut)
  },

  /**
   * Simply close the citation tool modal
   */
  closeModal: function () {
    this.$exportCitationTool.foundation('reveal', 'close')
  },

  /**
   * Hide the loader
   */
  hideLoader: function () {
    this.$loader.hide()
  },

  /**
   * Show the loader
   */
  showLoader: function () {
    this.$loader.show()
  },

  /**
   * Get product data and the formatted citation
   */
  getCitation: function () {
    const _self = this

    const cite = {
      citationStyle: _self.getSavedCitationStyle().key,
      initialised: _self.initialised,
      productIds: JSON.stringify(_self.productIds),
      citationStyleOptions: JSON.stringify(_self.citationStyleOptions)
    }

    $.post(_self.SERVICE_URL_CITE, cite).done(function (response) {
      if (!response.success) {
        _self.closeModal()
        _self.showMessage(_self.messages.error, { scroll: true, alertElement: $('#ajaxMessages') })
      } else {
        _self.updateInterface(response)
      }
    })
  },

  /**
   * Render the citation data in the scrolling div
   * @param citationData
   */
  renderCitationData: function (citationData) {
    const _self = this
    const citationDataDisplay = citationData.map(data => `<div>${data.citation}</div>`)

    _self.$exportCitationTool
      .find('#' + _self.CITATION_HTML_TEXT_ID)
      .html(citationDataDisplay)
      .scrollTop(0)

    _self.setClipboardSr()
  },

  setClipboardSr: function () {
    const _self = this
    const citationStyle = _self.getSavedCitationStyle()
    const citationName = citationStyle.displayName
    _self.$exportCitationTool
      .find('#' + _self.CITATION_CLIPBOARD_SR_INFO)
      .html('Citation clipboard, using selected format ' + citationName)
  },

  /**
   * Update the user interface with new data or an error message
   * @param err
   * @param data
   */
  updateInterface: function (data) {
    const _self = this
    _self.hideLoader()

    if (!_self.initialised) {
      // Populate modal with data and extra UI elements
      _self.$exportCitationTool.find('.content').html(data.html).show()
      _self.initEvents()
      _self.initialised = true
    }
    // Render the citation data
    _self.renderCitationData(data.citationData)
    AOP.focusTrap(this.CITATION_TOOL_MODAL_ID)
    _self.setAriaSelectedOption()
    $('#citation-modal').focus()
},

  /*
   * For any selected components, return the all matching reference data.
   * @return {array}
   */
  getProductIds: function () {
    const _self = this
    const selectedComponents = []
    _self.invalidProductsSelected = false
    $('input[name="productParts"]:checked').each(function () {
      if ($(this).is(':visible')) {
        if (_self.checkValidProductType(this)) {
          // Only add valid product types to our citation request list
          selectedComponents.push($(this).attr('data-prod-id'))
        } else {
          // Keep track if any invalid product types selected
          _self.invalidProductsSelected = true
        }
      }
    })
    return selectedComponents
  },

  /**
   * Check if we can generate a citation for the product type
   * @param item
   * @returns {boolean}
   */
  checkValidProductType: function (item) {
      return this.validProductTypes.indexOf($(item).data('prodType')) !== -1
  },

  /**
   * Check if two arrays contain the same values
   * @param arr1
   * @param arr2
   * @returns {boolean}
   */
  arraysEqual: function (arr1, arr2) {
    return JSON.stringify(arr1) === JSON.stringify(arr2)
  },

  /*
   * Show an error/info message
   * Wrapper for AOP.createAlertBox()
   */
  showMessage: function (errorMessage, opts) {
    opts = opts || {}
    opts.scroll = opts.scroll || false
    opts.alertType = opts.alertType || 'error'
    opts.alertElement =
      opts.alertElement ||
      (this.$exportCitationTool ? this.$exportCitationTool.find('#ajaxMessages') : $('#ajaxMessages'))
    AOP.createAlertBox(errorMessage || this.messages.error, opts)
  },

  /**
   * Clear any AOP alert messages
   */
  clearMessages: function () {
    this.$exportCitationTool.find('#ajaxMessages > .alert-box').remove()
  },

  /**
   * Test if the export citation button has been disable OR if the function has been disabled in the config
   * @param button
   * @returns {boolean}
   */
  checkEnabled: function (button) {
    const _self = this
    const $button = $(button)
    let enabled = true
    if ($button.hasClass('disabled')) {
      enabled = false
    }
    if ($button.attr('data-export-citation-disabled') === 'true') {
      enabled = false
      AOP.createAlertBox(_self.messages.invalidProductsSelected, { alertElement: $('#ajaxMessages') })
    }
    return enabled
  },

  /**
   * Reset the modal to loader state and create a new citation
   */
  loadNewCitation: async function () {
    const _self = this

    $('.' + _self.CITATION_UI_CLASS).hide()
    _self.showLoader()

      // Citation styles list
      $.get(_self.SERVICE_URL_CITE).done(function (response) {
        if (!response.success) {
          _self.closeModal()
          _self.showMessage(_self.messages.error, { scroll: true, alertElement: $('#ajaxMessages') })
        } else {
        _self.citationStyleOptions = response.citationStyleOptions

        _self.getCitation()
      }
    })
  },
    /**
   * Reset the modal to loader state and create a new citation
   */
  setAriaSelectedOption: function () {
    const _self = this

    const citationStyle = _self.getSavedCitationStyle()
    const citationStyleKey = citationStyle.key
    const $selectCitationStyle = _self.$selectCitationStyle
    $selectCitationStyle.find('option').each(function () {
      const $option = $(this)
      const optionKey = $option.val()
      $option.attr('aria-selected', optionKey === citationStyleKey)
    })
  },

  /**
   * Bind export citation buttons
   */
  initButtons: function () {
    const _self = this

    function doSingleCitation (button) {
      if (_self.checkEnabled(button)) {
        const productId = $(button).data('prodId')

        if (
          !_self.productIds.length ||
          _self.productIds.indexOf(productId) === -1
        ) {
          _self.initialised = false
          _self.productIds = [productId]
          _self.loadNewCitation()
        }

        _self.openModal()
      }
    }

    // Product page main export citation function
    $('.' + _self.PRODUCT_BUTTON_CLASS).on('click', function (e) {
      e.preventDefault()
      doSingleCitation(this)
    })

    // Single component in a listing
    $('body').on('click', '.' + _self.COMPONENT_BUTTON_CLASS, function (e) {
      e.preventDefault()
      doSingleCitation(this)
    })

    // All component displayed in listing
    $('body').on('click', '.' + _self.EXPORT_ALL_BUTTON, function (e) {
      e.preventDefault()
      _self.loadNewCitation()
    })

    // Multiple product citation
    $('.' + _self.COMPONENT_LIST_BUTTON_CLASS).on('click', function (e) {
      e.preventDefault()
      // Get all selected components
      const productIds = _self.getProductIds()
      // If only invalid product types have been selected, stop the citation and we can't generate one for the
      // selected type(s).
      if (_self.invalidProductsSelected && productIds.length === 0) {
        _self.showMessage(_self.messages.invalidProductsSelected, { scroll: true, alertElement: $('#ajaxMessages') })
        return
      }
      // Check we have something to cite
      if (productIds.length === 0) {
        _self.showMessage(_self.messages.noContent, { scroll: true, alertElement: $('#ajaxMessages') })
        return
      }

      // If there has been no change in the products selected, just open the modal.
      if (!_self.arraysEqual(productIds, _self.productIds)) {
        _self.productIds = productIds
        _self.initialised = false
        _self.loadNewCitation()
      }

      _self.openModal()
    })
  },

  initSelect: function () {
    const _self = this

    $('body').on('change', '.' + _self.SELECT_CITATION_STYLE_CLASS, function (e) {
      const citationStyleKey = e.target.value
      const citationStyleName = e.target.selectedOptions[0].text
      $('#announcement').text('Selected citation style: ' + citationStyleName)
      _self.saveCookie(citationStyleKey, citationStyleName)
      _self.getCitation()
      _self.setAriaSelectedOption()
    })
  },

  /**
   * If export citation tool has been turned off, bind all buttons to the relevant message.
   */
  disableTool: function () {
    const _self = this
    const buttons = [_self.PRODUCT_BUTTON_CLASS, _self.COMPONENT_LIST_BUTTON_CLASS, _self.COMPONENT_LIST_BUTTON_CLASS]
    $.each(buttons, function (key, button) {
      $('body').on('click', '.' + button, function (e) {
        e.preventDefault()
        _self.showMessage(_self.messages.citationToolSwitchOff, { scroll: true })
      })
    })
  },

  /**
   * Initialise the citation tool & bind button events.
   */
  init: function () {
    const _self = this
    $(document).ready(function () {
      // If export citation tool has been turned off, bind all buttons to the relevant message.
      if (!_self.SHOULD_USE_CITATION_TOOL) {
        _self.disableTool()
        return
      }
      // Create the modal and add to DOM
      _self.initContainer()

      _self.initButtons()
      _self.initSelect()
    })
  }
}

AOP.citationTool.init()
