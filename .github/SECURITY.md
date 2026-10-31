# Security Policy

The Advanced Production Hybrid RAG Engine team takes the security of our application, dependencies, and users seriously. We appreciate the efforts of security researchers and community members who help keep our project safe.

This document outlines our supported versions, vulnerability reporting process, and disclosure policies.

---

## Supported Versions

Security updates and patches are actively maintained for the following versions:

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| `main`  | :white_check_mark: | Active development & latest security patches |
| `< 1.0` | :x:                | Unsupported — please upgrade to `main` |

---

## Reporting a Security Vulnerability

> [!CAUTION]
> **Do NOT file public GitHub Issues, discussions, or pull requests for suspected security vulnerabilities.** Publicly disclosing vulnerabilities puts community users at risk before a fix can be made available.

### Preferred Method: GitHub Private Vulnerability Reporting
Please report security vulnerabilities privately using GitHub's built-in advisory tool:
1. Navigate to the repository's **[Security Tab](https://github.com/ShashankS1011/advanced-hybrid-rag/security)**.
2. Select **Advisories** on the left menu.
3. Click **Report a vulnerability** to open a confidential report directly to the repository maintainers.

### Alternative Method: Direct Security Contact
If you cannot use GitHub's advisory feature, please contact the repository maintainer directly via GitHub profile messaging or create a confidential advisory through GitHub's security portal.

---

## Information to Include in Your Report

To help us investigate, triage, and address the issue efficiently, please include as much of the following information as possible:

1. **Vulnerability Type:** (e.g., Remote Code Execution, Path Traversal, Arbitrary File Upload, Insecure Deserialization, Prompt Injection, Information Leakage).
2. **Component Affected:** (e.g., `src/api.py`, `src/ingest.py`, `src/retriever.py`, `app.py`, Docker configuration).
3. **Reproduction Steps:** Step-by-step instructions or a minimal Proof of Concept (PoC) demonstrating the vulnerability.
4. **Impact:** Realistic assessment of the vulnerability's impact and severity.
5. **Suggested Fix:** (Optional) If you have identified a fix or mitigation, feel free to share it.

---

## Response Process & Timelines

When a vulnerability report is submitted:

1. **Initial Acknowledgment:** We will acknowledge receipt of your report within **48 hours**.
2. **Triage & Validation:** We will assess and attempt to reproduce the reported vulnerability within **3 to 5 business days**, confirming its impact and severity.
3. **Resolution & Patching:** A fix will be developed in a private branch, verified against the test suite, and released promptly.
4. **Disclosure:** Once the patch is deployed and released, a security advisory will be published crediting the reporter (unless you prefer to remain anonymous).

---

## Security Best Practices for Deployments

If you are deploying this RAG engine in production or multi-user environments, we recommend the following precautions:

- **Secrets Management:** Never commit `.env` or hardcode `GROQ_API_KEY`. In cloud environments (AWS, GCP, Kubernetes), use dedicated secret managers.
- **Untrusted File Ingestion:** Validate and sanitize all uploaded PDFs. Run ingestion workflows within isolated, unprivileged containers to mitigate risks from malformed PDF parser exploits.
- **Network Boundaries:** Protect the Streamlit dashboard (`8501`) and FastAPI backend (`8000`) behind an authenticated reverse proxy (e.g., NGINX, Cloudflare Access, OAuth proxy) when exposing endpoints over public networks.
- **Vector Database Isolation:** Restrict file system permissions for the Chroma vector database storage directory (`CHROMA_DB_DIR`).

---

## Safe Harbor

We consider security research conducted under this policy to be authorized. We will not pursue legal action against researchers who:
- Make a good-faith effort to avoid privacy violations, data destruction, and service interruption.
- Report vulnerabilities to us promptly without public disclosure until a patch is released.
- Act in accordance with applicable laws and ethical disclosure guidelines.
