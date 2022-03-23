import os
from unittest import mock
from urllib.parse import urljoin

from atap_widgets.utils import _get_remote_jupyter_proxy_url
from atap_widgets.utils import _is_binder


def test_is_binder():
    """
    Test that _is_binder() returns true if BINDER_PORT is defined
    """
    with mock.patch.dict(os.environ, clear=True):
        assert not _is_binder()
    with mock.patch.dict(os.environ, {"BINDER_PORT": "'tcp://10.1.1.1:80'"}):
        assert _is_binder()


def test_get_remote_jupyter_proxy_url():
    eg_service_prefix = "/binder/jupyter/user/tester/"
    base_url = "https://test.org/"
    port = 123
    with mock.patch.dict(os.environ, {"JUPYTERHUB_SERVICE_PREFIX": eg_service_prefix}):
        # No port: return base_url host
        no_port_host = _get_remote_jupyter_proxy_url(port=None, base_url=base_url)
        assert no_port_host == "test.org"
        # With port: add jupyter prefix and proxy path
        with_port_url = _get_remote_jupyter_proxy_url(port=port, base_url=base_url)
        expected_url = urljoin(base_url, eg_service_prefix)
        expected_url = urljoin(expected_url, f"proxy/{port}")

        assert with_port_url == expected_url
