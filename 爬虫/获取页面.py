import urllib.request
import urllib.error
def askURL(url):
    html = None
    request = urllib.request.Request(url)
    try:
        response = urllib.request.urlopen(request)
        html = response.read()
        print(html)
    except urllib.error.URLError as e:
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
    return html
askURL("https://aistudio.baidu.com/projectDetail/101810")