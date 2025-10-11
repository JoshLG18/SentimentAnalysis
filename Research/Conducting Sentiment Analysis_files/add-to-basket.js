(function ($, AOP) {

  var ADD_TO_CART_ANCHORS = 'a[data-kk-action="addToCart"]';
  var CART_CONTAINER = '.ecommerce-cart';
  var $cartItemCountContainer = $('.no-of-items');
  var buyPrintCopyButton = $('.buy-print-copy-btn');
  var basketItemIds = new Array(); // jshint ignore:line
  var isLoaded = false;
  var $ecomAddToCartContainer = $('.book-ecommerce-overview-transactions-action-details');
  var $viewCartButton = $('a.format-view-cart');
  var $addToCartButton = $('a.format-add-to-cart');

  /**
   * Triggers an ajax request to the server to add an item to the basket
   *
   * @param productId
   * @param sku
   */
  function addToBasket(sku, productId) {

    function onSuccess() {
      if ($ecomAddToCartContainer.length > 0) {
        $('.alert-box').remove();
        AOP.createAlertBox('Item successfully added to the cart', {alertType: 'success'});
        $addToCartButton.removeClass('show');
        $viewCartButton.addClass('show');
      }
      
      var itemCount = getCartItemCount();
      itemCount = itemCount + 1;
      updateCartItemCount(itemCount);
      basketItemIds.push(sku);

      // Update the global header basket items count
      var $globalHeader = $('#global-header-wc');
      var currentCount = parseInt($globalHeader.attr('basket-items-count') || '0', 10);
      $globalHeader.attr('basket-items-count', currentCount + 1);
    }

    function onError (backendErrorMessage) {
      var $element = $('a[data-kk-action="addToCart"][data-sku="' + sku + '"]');
      var $inBasketMessage = $element.closest(CART_CONTAINER).find('.inBasket');
      var $error = $element.closest(CART_CONTAINER).find('.errorAdding');
      var errorMessage = backendErrorMessage;
      
      if (!errorMessage) {
        errorMessage = 'An error has occurred, please try again later.';
      }

      // Error handling for access modals in search listing
      if ($error.length) {
        $inBasketMessage.hide();
        $error.show();
        setTimeout(function restore () {
          $element.parent().show();
          $error.hide();
        }, 2000);
        return;
      }

      // Error handling for product landing page
      // Logic added to avoid duplication of error message in redesigned pages
      if ($element.data('showError')) {
        if ($('.alert-box').length > 0) {
          $('.alert-box').remove();
        }

        AOP.createAlertBox(errorMessage, {alertType: 'error'});
        $inBasketMessage.hide();
        $element.parent().show();
        return;
      }
    }

    return $.post(AOP.baseUrl + '/shopping-cart/add', {
      sku: sku,
      productId: productId
    }, onSuccess).fail(function (xhr) {
      var backendErrorMessage;
      try {
        var responseData = JSON.parse(xhr.responseText);
        backendErrorMessage = responseData.error;
      } catch (e) {
        backendErrorMessage = null;
      }
      
      onError(backendErrorMessage);
    });
  };

  /**
   * Helper function for different kinds of baskets (eg. adding books to cart, subscriptions)
   * @param {object} $self (jquery instiation object eg. $('#id'))
   * @returns nothing
   */
  function toggleAddToBasketDisplay ($self) {
    if (!$self) {
      return;
    }

    if ($ecomAddToCartContainer.length === 0) {
      $self.parent().hide();
    } else {
      $('button.digital.active').addClass('digital-in-cart');
    }

    $self.closest(CART_CONTAINER).find('.inBasket').show();
    return;
  }

  /**
   * Disable 'Add to cart' if item already in the basket
   */
  function labelAddedProducts () {
    var selector = $.map(basketItemIds, function (sku) {
      return new Array('a[data-kk-action="addToCart"][data-sku="' + sku + '"]').join(''); // jshint ignore:line
    }).join(',');

    $.each($(selector), function () {
      toggleAddToBasketDisplay($(this));
    });
  };

  /**
   * Executed once the document loads
   *
   * Will bind an ajax request to "Add to cart" buttons
   */
  function initializeAddToCartHandlers () {
    $(document).on('click keydown', ADD_TO_CART_ANCHORS, function (e) {
      if (e.type === 'keydown' && e.which !== 13 && e.which !== 32) {
        return;
      }
      e.preventDefault();
      var sku = $(this).data('sku');
      var productId = $(this).data('productId');
      var $element = $('a[data-kk-action="addToCart"][data-sku="' + sku + '"]');
      toggleAddToBasketDisplay($element);
      addToBasket(sku, productId);
      return false;
    });
  };

  /**
   * Returns number parsed from DOM
   *
   * @returns {Number}
   */
  function getCartItemCount () {
    // Call first() as there are 2 item counts - mobile and desktop.
    return parseInt($cartItemCountContainer.find('span').first().text());
  };

  /**
   * Updates cart item count in DOM
   *
   * @param value
   */
  function updateCartItemCount (value) {
    var visibility = !value ? 'hidden' : 'visible';
    var $mobileView = $('ul.show-for-small-only.profile-mobile');
    $mobileView.find('a').first().css('visibility', visibility);
    $cartItemCountContainer.html('( <span>' + value + '</span> )');
  };

  /**
   * Fetches count if cart items from the server.
   *
   * To be executed once the document loaded
   */
  function fetchCartItemCount () {
    $.get(AOP.baseUrl + '/shopping-cart/count', { _: $.now() }, function (data) {
      updateCartItemCount(data.totalItemCount);
      basketItemIds = data.itemIds;
      isLoaded = true;
      labelAddedProducts();
    });
  };

  /**
   * Present confirm Modal before buy print copy redirection
   */
  function initializePrintCopyModal () {
    buyPrintCopyButton.on('click', function (e) {
      e.stopPropagation();
      e.preventDefault();
      var url = $(this).attr('href');

      var fnConfirm = function () {
        window.open(url, '_blank');
      };

      var message = 'To complete your purchase of print-on-demand copies, you will be directed to Sheridan Custom Publishing. ' +
        'Any items added to your basket for purchase on the Cambridge Core site will remain in your basket and can be purchased ' +
        'separately from print-on-demand purchases.';
      AOP.confirmModal(null, message, fnConfirm, null, {confirmLabel: 'Proceed'});
    });
  };

  /**
   * Checks if product is in the basket
   */
  function isProductInBasket(id) {
    return basketItemIds.filter(function(item) { return item === id }).length > 0;
  };

  function isCartItemCountLoaded () {
    return isLoaded;
  };

  initializePrintCopyModal();
  fetchCartItemCount();
  initializeAddToCartHandlers();

  $(document).on('DOMSubtreeModified', '.accessible-alert', function() {
    labelAddedProducts();
  })

  AOP.basket = {
    addToBasket: addToBasket,
    isProductInBasket: isProductInBasket,
    isCartItemCountLoaded: isCartItemCountLoaded
  };
  /* global AOP */
})(jQuery, window.AOP || {});
