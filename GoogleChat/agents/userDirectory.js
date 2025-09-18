// Simple in-memory user directory with supervisor references and Chat spaces

// NOTE: In production, replace with a proper datastore.

// Relational mock data with fixed supervisor relations and Chat spaces.
// Override via env vars when available.
const users = new Map([
  [
    "user1@example.com",
    {
      email: "user1@example.com",
      name: "User One",
      spaceName: process.env.USER1_SPACE_NAME || null,
      supervisorEmail: "user2@example.com",
      supervisorSpaceName: process.env.USER2_SPACE_NAME || null,
    },
  ],
  [
    "user2@example.com",
    {
      email: "user2@example.com",
      name: "User Two (Supervisor)",
      spaceName: process.env.USER2_SPACE_NAME || null,
      supervisorEmail: null,
      supervisorSpaceName: null,
    },
  ],
  [
    "test@example.com",
    {
      email: "test@example.com",
      name: "Test User",
      spaceName: process.env.TEST_USER_SPACE_NAME || null,
      supervisorEmail:
        process.env.SUPERVISOR_EMAIL ||
        process.env.GOOGLE_CHAT_SUPERVISOR_EMAIL ||
        "user2@example.com",
      supervisorSpaceName:
        process.env.SUPERVISOR_SPACE_NAME ||
        process.env.USER2_SPACE_NAME ||
        null,
    },
  ],
]);

export function getUserProfile(userEmail) {
  if (!users.has(userEmail)) {
    users.set(userEmail, {
      email: userEmail,
      name: null,
      spaceName: null,
      supervisorEmail:
        process.env.SUPERVISOR_EMAIL ||
        process.env.GOOGLE_CHAT_SUPERVISOR_EMAIL ||
        null,
      supervisorSpaceName: null,
    });
  }
  return users.get(userEmail);
}

export function setSupervisorSpace(userEmail, spaceName) {
  const profile = getUserProfile(userEmail);
  profile.supervisorSpaceName = spaceName;
  users.set(userEmail, profile);
}

export function setSupervisorForUser(userEmail, supervisorEmail) {
  const profile = getUserProfile(userEmail);
  profile.supervisorEmail = supervisorEmail;
  users.set(userEmail, profile);
}

export function setUserSpace(userEmail, spaceName) {
  const profile = getUserProfile(userEmail);
  profile.spaceName = spaceName;
  users.set(userEmail, profile);
}

export function getSupervisorEmailFor(userEmail) {
  const profile = getUserProfile(userEmail);
  return profile?.supervisorEmail || null;
}

export function getSupervisorSpaceFor(userEmail) {
  const profile = getUserProfile(userEmail);
  return profile?.supervisorSpaceName || null;
}

export function getUserSpaceName(userEmail) {
  const profile = getUserProfile(userEmail);
  return profile?.spaceName || null;
}

export function listUsers() {
  return Array.from(users.values());
}
