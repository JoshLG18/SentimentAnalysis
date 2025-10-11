var AOP = AOP || {}

var countryTextInput = $('.country-picker .input-container input')
var updateButton = $('#eligibilityUpdateCountry')
var countryListUl = $('#countryList')
var dropdownIconBtn = $('#dropdownIcon')
var clearIconBtn = $('#clearIcon')
var errorMessageP = $('#countryPickerErrorMsg')
var legalNoticeMessageP = $('#countryPickerLegalNoticeMsg')
var countryLabel = $('.country-picker-label')

var sensitiveCountries = []
var validCountries = []
var dropdownIndex = -1

AOP.countryPickerEventListener = function (countries, callbacks) {
  sensitiveCountries = getSensitiveCountries(countries)
  validCountries = getValidCountries(countries)
  this.callbacks = callbacks || {}
  let isClicking = false

  $(countryTextInput).on('click', handleCountryDisplayToggle)

  $(dropdownIconBtn).on('click', handleCountryDisplayToggle)

  $(clearIconBtn).on('click', handleClearIconClick)

  $(countryTextInput).on('input', updateCountryInputState)

  $(countryListUl).on('mousedown', 'li', function () {
    isClicking = true
  })

  $(countryListUl).on('click', 'li', function () {
    isClicking = false
    var countryInfo = {
      value: $(this).data('country'),
      data: $(this).data('code')
    }

    handleCountrySelection(countryInfo, callbacks)
  })

  $(updateButton).on('click', function () {
    handleCountryInput(countryTextInput.val(), callbacks)
  })

  $(countryTextInput).on('keydown', function (e) {
    var countryList = countryListUl.find('li')

    handleKeyEvents(e, countryList, countryTextInput.val(), callbacks)
  })

  $(countryTextInput).on('blur', function () {
    if (isClicking) return
    handleCountryInput(countryTextInput.val(), callbacks)
  })
}

// Get the list of sensitive countries
function getSensitiveCountries (countries) {
  return countries.filter(function (country) {
    return country.isSensitiveTerritory
  })
}

// Filter list of countries without the list of sensitive countries
function getValidCountries (countries) {
  return countries.filter(function (country) {
    return !sensitiveCountries.some(function (sensitiveCountry) {
      return sensitiveCountry === country
    })
  })
}

// Display valid countries
function showCountries (countries) {
  countryListUl.empty()
  countries.forEach(function (country) {
    var li = $('<li>').text(`${country.data} (${country.value})`).attr({ 'data-code': country.data, 'data-country': country.value, 'aria-label': country.value })
    countryListUl.append(li)
  })
  countryListUl.show()
}

// Display countries based on input
function displayCountryInput () {
  var input = countryTextInput.val().toLowerCase()
  countryListUl.empty()

  if (input === '') {
    showCountries(validCountries)
    return
  }

  var countries = validCountries.filter(function (validCountry) {
    return validCountry.value.toLowerCase().includes(input) || validCountry.data.toLowerCase().includes(input)
  })

  countries.length > 0 ? showCountries(countries) : hideCountryList()
}

function countryWithLegalNotice (countryName, countryCode) {
  (countryName === 'Australia' || countryCode === 'AUS') ? legalNoticeMessageP.show() : legalNoticeMessageP.hide()
}

function hideCountryList () {
  countryListUl.hide()
  dropdownIndex = -1
}

function hideErrorMessage () {
  errorMessageP.hide()
  countryLabel.removeClass('error')
  countryTextInput.removeAttr('aria-describedby')
}

function showErrorMessage () {
  errorMessageP.show()
  countryLabel.addClass('error')
  countryTextInput.attr('aria-describedby', 'countryPickerErrorMsg')
}

function handleCallbacks (data, callbacks) {
  if (callbacks && typeof callbacks.loadInstitutionsData === 'function' && typeof callbacks.removeInstitutions === 'function') {
    callbacks.loadInstitutionsData(data)
    callbacks.removeInstitutions()
  }
}

function handleKeyEvents (e, countryList, countryInput, callbacks) {
  switch (e.key) {
    case 'Enter':
      e.preventDefault()
      handleCountryInput(countryInput, callbacks)
      break

    case 'ArrowDown':
      if (dropdownIndex < countryList.length - 1 && countryListUl.is(':visible')) {
        dropdownIndex++
        highlightCurrentCountry()
        countryList.eq(dropdownIndex)[0].scrollIntoView({ block: 'nearest' })
      }
      break

    case 'ArrowUp':
      if (dropdownIndex > 0 && countryListUl.is(':visible')) {
        dropdownIndex--
        highlightCurrentCountry()
        countryList.eq(dropdownIndex)[0].scrollIntoView({ block: 'nearest' })
      }
      break

    case 'Escape':
      hideCountryList()
      countryTextInput.val('')
      dropdownIconBtn.attr('aria-expanded', 'false')
      countryTextInput.focus()
      break

    default:
      break
  }
}

function highlightCurrentCountry () {
  var countryList = countryListUl.find('li')

  countryList.each(function (index) {
    $(this).removeClass('active')
    if (index === dropdownIndex) {
      $(this).addClass('active')
      countryTextInput.val($(this).data('code'))
    }
  })
}

function handleCountryDisplayToggle () {
  var isCountryListVisible = countryListUl.is(':visible')

  dropdownIconBtn.attr('aria-expanded', isCountryListVisible ? 'false' : 'true')
  isCountryListVisible ? hideCountryList() : showCountries(validCountries)
}

function handleClearIconClick () {
  countryTextInput.val('')
  countryTextInput.removeAttr('data-country')
  countryTextInput.removeAttr('data-code')
  hideCountryList()
  hideErrorMessage()
  clearIconBtn.hide()
  legalNoticeMessageP.hide()
}

function updateCountryInputState () {
  dropdownIndex = -1
  hideErrorMessage()
  displayCountryInput()
  clearIconBtn.css('display', $(this).val() ? 'block' : 'none')
}

function handleCountrySelection (countryInfo, callbacks) {
  countryTextInput.val(countryInfo.value)
  countryTextInput.attr({ 'data-country': countryInfo.value, 'data-code': countryInfo.data })
  clearIconBtn.show()
  hideCountryList()
  hideErrorMessage()
  countryWithLegalNotice(countryInfo.value, countryInfo.data)
  handleCallbacks(countryInfo, callbacks)
}

function handleSensitiveCountrySelection (countryInput, callbacks) {
  var sensitiveCountry = sensitiveCountries.find(function (country) {
    return country.value.toLowerCase().includes(countryInput.value.toLowerCase())
  })

  if (sensitiveCountry) {
    countryTextInput.val(sensitiveCountry.value)
    countryTextInput.attr({ 'data-country': sensitiveCountry.value, 'data-code': sensitiveCountry.data })
    legalNoticeMessageP.hide()
    hideCountryList()
    hideErrorMessage()
    handleCallbacks({ value: sensitiveCountry.value, data: sensitiveCountry.data }, callbacks)
    return
  }

  handleInvalidSelection()
}

function handleInvalidSelection () {
  clearIconBtn.show()
  legalNoticeMessageP.hide()
  showErrorMessage()
  hideCountryList()
  countryTextInput.removeAttr('data-country')
  countryTextInput.removeAttr('data-code')
}

function handleCountryInput (countryInput, callbacks) {
  var normalizedCountryInput = countryInput.toLowerCase()

  // Try to match by country name
  var countryMatch = validCountries.find(function (country) {
    return country.value.toLowerCase() === normalizedCountryInput
  })

  // Try to match by ISO code if no name match found
  if (!countryMatch) {
    countryMatch = validCountries.find(function (country) {
      return country.data.toLowerCase() === normalizedCountryInput
    })
  }

  // Handle exact match
  if (countryMatch) {
    handleCountrySelection(countryMatch, callbacks)
    return
  }

  // Handle partial matches for valid countries
  var multipleValidCountryMatches = validCountries.filter(function (country) {
    return country.value.toLowerCase().indexOf(normalizedCountryInput) !== -1 || country.data.toLowerCase().indexOf(normalizedCountryInput) !== -1
  })

  // Handle partial matches for sensitive countries
  var multipleSensitiveCountryMatches = sensitiveCountries.filter(function (country) {
    return country.value.toLowerCase().indexOf(normalizedCountryInput) !== -1
  })

  // Handle multiple matches
  if (multipleValidCountryMatches.length > 1 || multipleSensitiveCountryMatches.length > 1) {
    handleInvalidSelection()
    return
  }

  if (multipleValidCountryMatches.length === 1 && multipleSensitiveCountryMatches.length === 0) {
    handleCountrySelection(multipleValidCountryMatches[0], callbacks)
    return
  }

  if (multipleSensitiveCountryMatches.length === 1 && multipleValidCountryMatches.length === 0) {
    handleSensitiveCountrySelection(multipleSensitiveCountryMatches[0], callbacks)
    return
  }

  handleInvalidSelection()
}
