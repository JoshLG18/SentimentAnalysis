(function () {
  function revealModalMod (e) {
    e.preventDefault()
    e.stopPropagation()
    var $targetDom = $('#' + $(this).data('reveal-id')).first()

    $targetDom.foundation('reveal', 'open')
    $targetDom.on('open.fndtn.reveal', function (event) {
      if (event.namespace !== 'fndtn.reveal') {

      }
    })
  }

  function bindRevealModal (event) {
    $('[data-reveal-id]').off('click').on('click', revealModalMod)

    // .off above prevents initial click to load modal, force here
    if (event) {
      $(event.target).click()
    }
  }
  bindRevealModal()
  $(document).on('click', '[data-reveal-id]', bindRevealModal)
})()
