import express from "express";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();
const router = express.Router();

/**
 * GET /api/extension/status
 * Simple auth check for the extension. Adjust the user detection to your real auth.
 */
router.get("/status", (req, res) => {
  const isAuthed =
    !!req.user ||
    !!req.session?.user ||
    !!req.cookies?.sid ||
    !!req.cookies?.token;

  if (!isAuthed) return res.status(401).json({ ok: false });
  res.json({ ok: true });
});

/**
 * GET /api/extension/autofill
 * Return data the extension will inject. Replace with real DB queries later.
 */
router.get("/autofill", async (_req, res) => {
  // Example: read something from DB and return it
  // const latest = await prisma.vehicle.findMany({ take: 5, orderBy: { createdAt: "desc" }});
  res.json({ ok: true, data: { sample: "autofill data" } });
});

/**
 * POST /api/extension/vehicle
 * Example – save a vehicle coming from the extension.
 */
router.post("/vehicle", async (req, res) => {
  try {
    const { plate, model, notes } = req.body ?? {};
    const v = await prisma.vehicle.create({ data: { plate, model, notes } });
    res.json({ ok: true, vehicle: v });
  } catch (e) {
    console.error(e);
    res.status(500).json({ ok: false, error: "Failed to save vehicle" });
  }
});

export default router;