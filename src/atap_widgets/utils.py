import os
import urllib


DEFAULT_MYBINDER_HOST = "https://hub.gke.mybinder.org"


def _get_remote_jupyter_proxy_url(port: int, base_url: str = None):
    """
    Callable to configure Bokeh's show method when a proxy must be
    configured. If the environment variable BINDER_EXTERNAL_URL
    is set, this will be used as the base_url. Otherwise
    we fall back to DEFAULT_MYBINDER_HOST.

    If port is None we're asking about the URL
    for the origin header.
    """
    if base_url is None:
        if "BINDER_EXTERNAL_URL" in os.environ:
            base_url = os.environ["BINDER_EXTERNAL_URL"]
            if not base_url.startswith("https://"):
                # The "https://" was removed from the ATAP binderhub since early 2025, hence need to add it back manually
                base_url = "https://" + base_url
        else:
            base_url = DEFAULT_MYBINDER_HOST
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
