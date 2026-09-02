"""Tests for the mcp.py module in serena."""

import asyncio
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools.base import Tool as MCPTool

from serena.agent import Tool, ToolRegistry
from serena.config.context_mode import SerenaAgentContext
from serena.mcp import (
    SerenaFastMCP,
    SerenaMCPFactory,
)
from serena.tools import ToolMarkerCanEdit

make_tool = SerenaMCPFactory.make_mcp_tool


# Create a mock agent for tool initialization
class MockAgent:
    def __init__(self):
        self.project_config = None
        self.serena_config = None

    @staticmethod
    def get_context() -> SerenaAgentContext:
        return SerenaAgentContext.load_default()


class BaseMockTool(Tool):
    """A mock Tool class for testing."""

    def __init__(self):
        super().__init__(MockAgent())


class BasicTool(BaseMockTool):
    """A mock Tool class for testing."""

    def apply(self, name: str, age: int = 0) -> str:
        """This is a test function.

        :param name: The person's name
        :param age: The person's age
        :return: A greeting message
        """
        return f"Hello {name}, you are {age} years old!"

    def apply_ex(
        self,
        log_call: bool = True,
        catch_exceptions: bool = True,
        **kwargs,
    ) -> str:
        """Mock implementation of apply_ex."""
        kwargs.pop("mcp_ctx", None)
        return self.apply(**kwargs)


class EditingTool(BasicTool, ToolMarkerCanEdit):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def apply(self, name: str, age: int = 0) -> str:
        """Record an editing call and return a greeting."""
        self.call_count += 1
        return super().apply(name=name, age=age)


class TokenVerifierStub:
    async def verify_token(self, token: str) -> AccessToken | None:
        return None


class ScopedTokenVerifierStub:
    async def verify_token(self, token: str) -> AccessToken | None:
        scopes = {
            "reader": ["serena:read"],
            "writer": ["serena:write"],
        }.get(token)
        if scopes is None:
            return None
        return AccessToken(
            token=token,
            client_id="shared-client",
            scopes=scopes,
            subject="shared-subject",
            claims={"iss": "https://broker.example.com"},
        )


AUTH_SETTINGS = AuthSettings(
    issuer_url="https://broker.example.com",
    resource_server_url="http://127.0.0.1:8000",
)
TOKEN_VERIFIER = TokenVerifierStub()


@contextmanager
def access_token_scope(*scopes: str) -> Iterator[None]:
    access_token = AccessToken(token="token", client_id="test-client", scopes=list(scopes))
    context_token = auth_context_var.set(AuthenticatedUser(access_token))
    try:
        yield
    finally:
        auth_context_var.reset(context_token)


def make_authorized_server(
    *,
    token_verifier: TokenVerifierStub | ScopedTokenVerifierStub = TOKEN_VERIFIER,
    json_response: bool = False,
) -> tuple[SerenaFastMCP, EditingTool]:
    read_tool = make_tool(BasicTool())
    editing_tool = EditingTool()
    edit_tool = make_tool(editing_tool)
    server = SerenaFastMCP(
        name="Test Serena",
        token_verifier=token_verifier,
        auth=AUTH_SETTINGS,
        json_response=json_response,
    )
    server._tool_manager._tools = {
        read_tool.name: read_tool,
        edit_tool.name: edit_tool,
    }
    return server, editing_tool


def mcp_http_headers(token: str, session_id: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
    }
    if session_id is not None:
        headers["Mcp-Session-Id"] = session_id
    return headers


def initialize_mcp_http_session(client, token: str) -> str:
    response = client.post(
        "/mcp",
        headers=mcp_http_headers(token),
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "authorization-test", "version": "1.0"},
            },
        },
    )
    assert response.status_code == 200
    session_id = response.headers["mcp-session-id"]
    initialized = client.post(
        "/mcp",
        headers=mcp_http_headers(token, session_id),
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert initialized.status_code == 202
    return session_id


@pytest.mark.parametrize(
    ("scopes", "expected_tools"),
    [
        (("serena:read",), {"basic"}),
        (("serena:write",), {"basic", "editing"}),
        (("unrelated",), set()),
        ((), set()),
    ],
)
def test_authenticated_tool_discovery_uses_serena_scopes(scopes: tuple[str, ...], expected_tools: set[str]) -> None:
    server, _ = make_authorized_server()

    with access_token_scope(*scopes):
        tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == expected_tools


def test_reader_direct_call_rejects_editing_tool_before_execution() -> None:
    server, editing_tool = make_authorized_server()

    with access_token_scope("serena:read"):
        with pytest.raises(ToolError, match="reader connection cannot call editing tool"):
            asyncio.run(server.call_tool("editing", {"name": "Reader"}))

    assert editing_tool.call_count == 0


def test_reader_can_directly_call_non_editing_tool() -> None:
    server, _ = make_authorized_server()

    with access_token_scope("serena:read"):
        asyncio.run(server.call_tool("basic", {"name": "Reader"}))


def test_writer_can_directly_call_editing_tool() -> None:
    server, editing_tool = make_authorized_server()

    with access_token_scope("serena:write"):
        asyncio.run(server.call_tool("editing", {"name": "Writer"}))

    assert editing_tool.call_count == 1


def test_server_without_auth_preserves_existing_tool_access() -> None:
    server = SerenaFastMCP(name="Test Serena")
    read_tool = make_tool(BasicTool())
    edit_tool = make_tool(EditingTool())
    server._tool_manager._tools = {read_tool.name: read_tool, edit_tool.name: edit_tool}

    tools = asyncio.run(server.list_tools())
    asyncio.run(server.call_tool("editing", {"name": "Writer"}))

    assert {tool.name for tool in tools} == {"basic", "editing"}


def test_authenticated_server_without_request_token_denies_tool_access() -> None:
    server, editing_tool = make_authorized_server()

    assert asyncio.run(server.list_tools()) == []
    with pytest.raises(ToolError, match="bearer token does not grant access"):
        asyncio.run(server.call_tool("editing", {"name": "Unknown"}))
    assert editing_tool.call_count == 0


def test_sdk_authentication_rejects_missing_and_invalid_bearer_tokens() -> None:
    server, _ = make_authorized_server()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated")
        from starlette.testclient import TestClient

        client = TestClient(server.streamable_http_app())

        missing_token = client.post("/mcp")
        invalid_token = client.post("/mcp", headers={"Authorization": "Bearer invalid"})

    assert missing_token.status_code == 401
    assert invalid_token.status_code == 401


def test_stateful_http_uses_reader_scope_from_each_request() -> None:
    server, editing_tool = make_authorized_server(
        token_verifier=ScopedTokenVerifierStub(),
        json_response=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated")
        from starlette.testclient import TestClient

        with TestClient(server.streamable_http_app(), base_url="http://localhost:8000") as client:
            session_id = initialize_mcp_http_session(client, "writer")
            list_response = client.post(
                "/mcp",
                headers=mcp_http_headers("reader", session_id),
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            call_response = client.post(
                "/mcp",
                headers=mcp_http_headers("reader", session_id),
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "editing", "arguments": {"name": "Reader"}},
                },
            )

    assert list_response.status_code == 200
    assert {tool["name"] for tool in list_response.json()["result"]["tools"]} == {"basic"}
    assert call_response.status_code == 200
    assert "reader connection cannot call editing tool" in call_response.text.lower()
    assert editing_tool.call_count == 0


def test_stateful_http_uses_writer_scope_from_each_request() -> None:
    server, editing_tool = make_authorized_server(
        token_verifier=ScopedTokenVerifierStub(),
        json_response=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated")
        from starlette.testclient import TestClient

        with TestClient(server.streamable_http_app(), base_url="http://localhost:8000") as client:
            session_id = initialize_mcp_http_session(client, "reader")
            list_response = client.post(
                "/mcp",
                headers=mcp_http_headers("writer", session_id),
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            call_response = client.post(
                "/mcp",
                headers=mcp_http_headers("writer", session_id),
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "editing", "arguments": {"name": "Writer"}},
                },
            )

    assert list_response.status_code == 200
    assert {tool["name"] for tool in list_response.json()["result"]["tools"]} == {"basic", "editing"}
    assert call_response.status_code == 200
    assert not call_response.json()["result"]["isError"]
    assert editing_tool.call_count == 1


def test_factory_requires_verifier_and_auth_settings_together() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        SerenaMCPFactory(transport="streamable-http", token_verifier=TOKEN_VERIFIER)

    with pytest.raises(ValueError, match="must be provided together"):
        SerenaMCPFactory(transport="streamable-http", auth_settings=AUTH_SETTINGS)


def test_factory_rejects_global_required_scopes() -> None:
    auth_settings = AUTH_SETTINGS.model_copy(update={"required_scopes": ["serena:read"]})

    with pytest.raises(ValueError, match="required_scopes"):
        SerenaMCPFactory(
            transport="streamable-http",
            token_verifier=TOKEN_VERIFIER,
            auth_settings=auth_settings,
        )


def test_factory_accepts_verifier_and_auth_settings() -> None:
    factory = SerenaMCPFactory(
        transport="streamable-http",
        token_verifier=TOKEN_VERIFIER,
        auth_settings=AUTH_SETTINGS,
    )

    assert factory.token_verifier is TOKEN_VERIFIER
    assert factory.auth_settings is AUTH_SETTINGS


def test_make_tool_basic() -> None:
    """Test that make_tool correctly creates an MCP tool from a Tool object."""
    mock_tool = BasicTool()

    mcp_tool = make_tool(mock_tool)

    # Test that the MCP tool has the correct properties
    assert isinstance(mcp_tool, MCPTool)
    assert mcp_tool.name == "basic"
    assert "This is a test function. Returns A greeting message." in mcp_tool.description

    # Test that the parameters were correctly processed
    parameters = mcp_tool.parameters
    assert "properties" in parameters
    assert "name" in parameters["properties"]
    assert "age" in parameters["properties"]
    assert parameters["properties"]["name"]["description"] == "The person's name."
    assert parameters["properties"]["age"]["description"] == "The person's age."


def test_make_tool_execution() -> None:
    """Test that the execution function created by make_tool works correctly."""
    mock_tool = BasicTool()
    mcp_tool = make_tool(mock_tool)

    # Execute the MCP tool function
    result = mcp_tool.fn(name="Alice", age=30)

    assert result == "Hello Alice, you are 30 years old!"


def test_make_tool_no_params() -> None:
    """Test make_tool with a function that has no parameters."""

    class NoParamsTool(BaseMockTool):
        def apply(self) -> str:
            """This is a test function with no parameters.

            :return: A simple result
            """
            return "Simple result"

        def apply_ex(self, *args, **kwargs) -> str:
            return self.apply()

    tool = NoParamsTool()
    mcp_tool = make_tool(tool)

    assert mcp_tool.name == "no_params"
    assert "This is a test function with no parameters. Returns A simple result." in mcp_tool.description
    assert mcp_tool.parameters["properties"] == {}


def test_make_tool_no_return_description() -> None:
    """Test make_tool with a function that has no return description."""

    class NoReturnTool(BaseMockTool):
        def apply(self, param: str) -> str:
            """This is a test function.

            :param param: The parameter
            """
            return f"Processed: {param}"

        def apply_ex(self, *args, **kwargs) -> str:
            return self.apply(**kwargs)

    tool = NoReturnTool()
    mcp_tool = make_tool(tool)

    assert mcp_tool.name == "no_return"
    assert mcp_tool.description == "This is a test function."
    assert mcp_tool.parameters["properties"]["param"]["description"] == "The parameter."


def test_make_tool_parameter_not_in_docstring() -> None:
    """Test make_tool when a parameter in properties is not in the docstring."""

    class MissingParamTool(BaseMockTool):
        def apply(self, name: str, missing_param: str = "") -> str:
            """This is a test function.

            :param name: The person's name
            """
            return f"Hello {name}! Missing param: {missing_param}"

        def apply_ex(self, *args, **kwargs) -> str:
            return self.apply(**kwargs)

    tool = MissingParamTool()
    mcp_tool = make_tool(tool)

    assert "name" in mcp_tool.parameters["properties"]
    assert "missing_param" in mcp_tool.parameters["properties"]
    assert mcp_tool.parameters["properties"]["name"]["description"] == "The person's name."
    assert "description" not in mcp_tool.parameters["properties"]["missing_param"]


def test_make_tool_multiline_docstring() -> None:
    """Test make_tool with a complex multi-line docstring."""

    class ComplexDocTool(BaseMockTool):
        def apply(self, project_file_path: str, host: str, port: int) -> str:
            """Create an MCP server.

            This function creates and configures a Model Context Protocol server
            with the specified settings.

            :param project_file_path: The path to the project file, or None
            :param host: The host to bind to
            :param port: The port to bind to
            :return: A configured FastMCP server instance
            """
            return f"Server config: {project_file_path}, {host}:{port}"

        def apply_ex(self, *args, **kwargs) -> str:
            return self.apply(**kwargs)

    tool = ComplexDocTool()
    mcp_tool = make_tool(tool)

    assert "Create an MCP server" in mcp_tool.description
    assert "Returns A configured FastMCP server instance" in mcp_tool.description
    assert mcp_tool.parameters["properties"]["project_file_path"]["description"] == "The path to the project file, or None."
    assert mcp_tool.parameters["properties"]["host"]["description"] == "The host to bind to."
    assert mcp_tool.parameters["properties"]["port"]["description"] == "The port to bind to."


def test_make_tool_capitalization_and_periods() -> None:
    """Test that make_tool properly handles capitalization and periods in descriptions."""

    class FormatTool(BaseMockTool):
        def apply(self, param1: str, param2: str, param3: str) -> str:
            """Test function.

            :param param1: lowercase description
            :param param2: description with period.
            :param param3: description with Capitalized word.
            """
            return f"Formatted: {param1}, {param2}, {param3}"

        def apply_ex(self, *args, **kwargs) -> str:
            return self.apply(**kwargs)

    tool = FormatTool()
    mcp_tool = make_tool(tool)

    assert mcp_tool.parameters["properties"]["param1"]["description"] == "Lowercase description."
    assert mcp_tool.parameters["properties"]["param2"]["description"] == "Description with period."
    assert mcp_tool.parameters["properties"]["param3"]["description"] == "Description with Capitalized word."


def test_make_tool_missing_apply() -> None:
    """Test make_tool with a tool that doesn't have an apply method."""

    class BadTool(BaseMockTool):
        pass

    tool = BadTool()

    with pytest.raises(AttributeError):
        make_tool(tool)


@pytest.mark.parametrize(
    "docstring, expected_description",
    [
        (
            """This is a test function.

            :param param: The parameter
            :return: A result
            """,
            "This is a test function. Returns A result.",
        ),
        (
            """
            :param param: The parameter
            :return: A result
            """,
            "Returns A result.",
        ),
        (
            """
            :param param: The parameter
            """,
            "",
        ),
        ("Description without params.", "Description without params."),
    ],
)
def test_make_tool_descriptions(docstring, expected_description) -> None:
    """Test make_tool with various docstring formats."""

    class TestTool(BaseMockTool):
        def apply(self, param: str) -> str:
            return f"Result: {param}"

        def apply_ex(self, *args, **kwargs) -> str:
            return self.apply(**kwargs)

    # Dynamically set the docstring
    TestTool.apply.__doc__ = docstring

    tool = TestTool()
    mcp_tool = make_tool(tool)

    assert mcp_tool.name == "test"
    assert mcp_tool.description == expected_description


def is_test_mock_class(tool_class: type) -> bool:
    """Check if a class is a test mock class."""
    # Check if the class is defined in a test module
    module_name = tool_class.__module__
    return (
        module_name.startswith(("test.", "tests."))
        or "test_" in module_name
        or tool_class.__name__
        in [
            "BaseMockTool",
            "BasicTool",
            "BadTool",
            "NoParamsTool",
            "NoReturnTool",
            "MissingParamTool",
            "ComplexDocTool",
            "FormatTool",
            "NoDescriptionTool",
        ]
    )


@pytest.mark.parametrize("tool_class", ToolRegistry().get_all_tool_classes())
def test_make_tool_all_tools(tool_class) -> None:
    """Test that make_tool works for all tools in the codebase."""
    # Create an instance of the tool
    tool_instance = tool_class(MockAgent())

    # Try to create an MCP tool from it
    mcp_tool = make_tool(tool_instance)

    # Basic validation
    assert isinstance(mcp_tool, MCPTool)
    assert mcp_tool.name == tool_class.get_name_from_cls()

    # The description should be a string (either from docstring or default)
    assert isinstance(mcp_tool.description, str)
