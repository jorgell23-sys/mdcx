# Security

## What this package protects, and what it does not

An `.mdcx` file is encrypted at rest with AES-256-GCM and decrypted into memory
when opened. The key is derived from a passphrase with scrypt. Altering a single
byte of the body makes decryption fail rather than return altered data, so a
corrupted or tampered package is refused instead of silently trusted.

This is not searchable encryption. Querying a package decrypts it in memory, so
anyone who can read the memory of the process can read the corpus. A package is
protected in transit and at rest, not from the machine that opens it.

The passphrase is the whole of the protection. A weak one circulates as easily
as the file does, and there is no recovery: a package whose passphrase is lost
cannot be opened by anyone, including whoever built it.

## Reporting a vulnerability

Report privately through GitHub's advisory form, which does not make the report
public while it is being looked at:

https://github.com/jorgell23-sys/mdcx/security/advisories/new

Please do not open a public issue for something that affects the confidentiality
or the integrity of a package.

What helps most in a report: the version, what an attacker would be able to do,
and the smallest thing that demonstrates it. A package that reproduces the
problem is more useful than a description of one, provided it holds nothing
confidential.

## Supported versions

Fixes go to the current release. The version number is in `mdcx --version` and
in the package metadata; there is no long-term support branch.
