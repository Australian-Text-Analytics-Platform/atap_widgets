import os
import urllib


def _get_remote_jupyter_proxy_url(
    port: int, base_url: str = "https://notebooks.gesis.org/"
):
    """
    Callable to configure Bokeh's show method when a proxy must be
    configured.

    If port is None we're asking about the URL
    for the origin header.
    """
    host = urllib.parse.urlparse(base_url).netloc

    # If port is None we're asking for the URL origin
    # so return the public hostname.
    if port is None:
        return host

    service_url_path = os.environ["JUPYTERHUB_SERVICE_PREFIX"]
    proxy_url_path = "proxy/%d" % port

    user_url = urllib.parse.urljoin(base_url, service_url_path)
    full_url = urllib.parse.urljoin(user_url, proxy_url_path)
    return full_url


def _is_binder() -> bool:
    """
    Check if we are running in a Binder environment - this
    may affect how we interact with libraries like bokeh
    """
    return os.environ.get("BINDER_PORT") is not None
