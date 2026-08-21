# Publishing mdcx to PyPI

Everything is prepared. Two things need a person: creating the account and
generating a token. The rest is one command.

## Why publish here

Today someone has to clone the repository and point pip at the folder. After
publishing, anyone runs:

    pip install mdcx

It is also a prerequisite for the MCP Registry, which stores only metadata and
requires the package to exist in an index first.

## Step 1: create the account

Go to https://pypi.org/account/register/ and create an account. Two-factor
authentication is mandatory; an authenticator application on the phone is enough.

## Step 2: generate a token

At https://pypi.org/manage/account/token/ create an API token.

Scope: choose "Entire account" for the first upload. The project does not exist
yet, so a project-scoped token cannot be created until after the first release.
Once uploaded, replace it with a token scoped to `mdcx` alone and delete the
account-wide one.

The token is shown once. It begins with `pypi-`.

## Step 3: test upload (recommended)

TestPyPI is a separate copy of PyPI meant for rehearsal. It needs its own account
at https://test.pypi.org/account/register/ and its own token.

    cd publicar
    ..\.venv\Scripts\python.exe -m twine upload --repository testpypi dist/*

It will ask for a username, which is `__token__` literally, and the token as
password.

Then check the result at https://test.pypi.org/project/mdcx/ and install it in a
clean environment:

    pip install --index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ "mdcx[mcp]"

The second index is required because TestPyPI does not host the dependencies.

## Step 4: real upload

    cd publicar
    ..\.venv\Scripts\python.exe -m twine upload dist/*

Same credentials: `__token__` as username, the token as password.

## What is uploaded

    dist/mdcx-1.0.0-py3-none-any.whl    the wheel, what pip installs
    dist/mdcx-1.0.0.tar.gz              the source distribution, which anyone
                                        can inspect and rebuild

Both passed `twine check`. Verified by installing the wheel in a clean
interpreter: the three commands appear, and packing, signing and verification
work.

## Irreversible

A version published to PyPI cannot be replaced. It can be deleted, but that
number can never be reused: if 1.0.0 has a defect, the fix is 1.0.1.

This is why the rehearsal on TestPyPI is worth the extra ten minutes.

## After publishing

The README installation table becomes true as written. Then the MCP Registry
becomes possible, which is what puts the server in the catalogue that Claude,
ChatGPT and the other clients consult.

## Step 5: the MCP Registry

Once the package is on PyPI, the server can be listed in the official registry at
`registry.modelcontextprotocol.io`, which is the catalogue MCP clients consult to
discover available servers.

The registry stores metadata only, which is why PyPI comes first. Ownership is
verified by looking for a marker inside the package description: this repository
carries `<!-- mcp-name: io.github.jorgell23-sys/markdown-document-search -->` at the top of the
README, and `server.json` declares the same name. Both must match, and the marker
must already be in the published version: adding it later would require a new
release.

Publishing uses the official CLI:

    mcp-publisher login github
    mcp-publisher publish

The `io.github.jorgell23-sys/*` namespace is granted by authenticating with that
GitHub account, so no separate ownership claim is needed.

The registry is in preview: breaking changes or data resets may occur before
general availability.
