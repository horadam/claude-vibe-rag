# Information Security Policy

## Overview

Northwind Robotics handles sensitive customer data, proprietary robotics designs, and regulated financial information. All employees, contractors, and vendors with access to company systems must comply with this policy.

## Device Policy

- Only company-issued devices may be used to access production systems, internal source code repositories, or any system containing customer data.
- Personal devices may be used for company email and calendar via approved mobile apps (Outlook, Google Workspace mobile) only.
- Employees must enable full-disk encryption on company-issued laptops. IT verifies encryption compliance quarterly.
- Lost or stolen devices must be reported to IT within **24 hours** so the device can be remotely wiped.

## Network Security

- Employees must use the company VPN whenever accessing internal systems from outside a Northwind Robotics office.
- Public Wi-Fi (coffee shops, airports, hotels) must never be used without VPN active.
- Home networks must have WPA2 or WPA3 encryption enabled on the router.

This requirement applies to remote employees at all times and to on-site employees when traveling. See the Remote and Hybrid Work Policy for additional remote-work requirements.

## Access Control

- Access to systems and data is granted on a least-privilege basis.
- Employees must not share login credentials under any circumstances.
- Multi-factor authentication (MFA) is required on all company accounts. Exceptions require CISO approval.
- Access is revoked within **4 hours** of an employee's last day. See the Termination and Offboarding Policy for the full offboarding checklist.

## Data Handling

- Customer data may not be stored on personal devices, personal cloud storage, or any unapproved third-party service.
- Sensitive data at rest must be encrypted. AES-256 is the minimum standard.
- Production database access requires approval from the relevant team lead and is logged.

## Incident Reporting

Security incidents — including phishing attempts, suspected malware, unauthorized access, or data exposure — must be reported to **security@northwindrobotics.example** within **1 hour** of discovery.

Employees who discover or suspect an incident should:
1. Disconnect the affected device from the network (do not power off).
2. Report immediately to the security team.
3. Preserve logs and do not attempt remediation without guidance from IT.

## Consequences of Non-Compliance

Violations of this policy are subject to disciplinary action under the Code of Conduct, up to and including termination. Willful violations involving customer data may also result in legal action.
