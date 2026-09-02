from mcp.server.auth.provider import AccessToken
from mcp.server.fastmcp.exceptions import ToolError


class SerenaToolAuthorization:
    """Authorize Serena tool discovery and execution from bearer-token scopes."""

    READ_SCOPE = "serena:read"
    WRITE_SCOPE = "serena:write"

    @classmethod
    def can_access(cls, *, can_edit: bool, access_token: AccessToken | None) -> bool:
        """
        Return whether the access token permits use of a tool.

        :param can_edit: whether the tool can edit project state
        :param access_token: the verified bearer-token metadata, if available
        """
        if access_token is None:
            return False
        scopes = set(access_token.scopes)
        if cls.WRITE_SCOPE in scopes:
            return True
        return not can_edit and cls.READ_SCOPE in scopes

    @classmethod
    def require_access(cls, *, tool_name: str, can_edit: bool, access_token: AccessToken | None) -> None:
        """
        Require the access token to permit execution of a tool.

        :param tool_name: the exposed MCP tool name
        :param can_edit: whether the tool can edit project state
        :param access_token: the verified bearer-token metadata, if available
        """
        if cls.can_access(can_edit=can_edit, access_token=access_token):
            return
        if access_token is not None and cls.READ_SCOPE in access_token.scopes and can_edit:
            raise ToolError(f"A reader connection cannot call editing tool '{tool_name}'.")
        raise ToolError(f"The bearer token does not grant access to tool '{tool_name}'.")
