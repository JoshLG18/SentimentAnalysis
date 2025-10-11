// globals window
var AOP = AOP || {}

AOP.parseQuerystring = function (querystring) {
  querystring = querystring || (window.location.search)
  if (!querystring) {
    return {}
  }
  var match
    var pl = /\+/g // Regex for replacing addition symbol with a space
    var search = /([^&=]+)=?([^&]*)/g
    var decode = function (s) {
      return decodeURIComponent(s.replace(pl, ' '))
    }
    var query = querystring.substring(1)
    var urlParams = {}
  while (match = search.exec(query)) {
    var m1 = decode(match[1])
    if (urlParams[m1]) {
      if (!AOP.isArray(urlParams[m1])) {
        urlParams[m1] = [urlParams[m1]]
      }
      urlParams[m1].push(decode(match[2]))
    } else {
      urlParams[m1] = decode(match[2])
    }
  }
  return urlParams
}
