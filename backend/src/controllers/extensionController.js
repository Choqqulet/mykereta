import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

/**
 * Normalize ISO date (YYYY-MM-DD) or return "" if null.
 */
function toISODate(d) {
  try {
    return d ? new Date(d).toISOString().slice(0, 10) : "";
  } catch {
    return "";
  }
}

/**
 * GET /api/extension/autofill
 * Auth: cookie-based (mykereta_session) → requireUser middleware should populate req.user.id
 */
export async function getAutofillBundle(req, res, next) {
  try {
    // If the middleware uses a different property, adjust here.
    const userId = req.user?.id;
    if (!userId) return res.status(401).json({ ok: false, error: "UNAUTHENTICATED" });

    // User fields per schema
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        name: true,
        avatarUrl: true,
      },
    });
    if (!user) return res.status(401).json({ ok: false, error: "USER_NOT_FOUND" });

    // Most-recent vehicle by updatedAt (present in your Vehicle model)
    const vehicle = await prisma.vehicle.findFirst({
      where: { userId },
      orderBy: { updatedAt: "desc" },
      select: {
        brand: true,
        model: true,
        year: true,
        plate: true,
        color: true,
        fuelType: true,
        chassisNumber: true,
        engineNumber: true,
        roadTaxExpiry: true,
        insuranceExpiry: true,
        lastServiceDate: true,
        nextServiceDate: true,
        currentMileage: true,
      },
    });

    // Shape the flat, extension-friendly payload
    const fields = {
      // User basics
      fullName: user.name ?? "",
      email: user.email ?? "",
      avatarUrl: user.avatarUrl ?? "",

      // Vehicle (if any)
      vehicleBrand: vehicle?.brand ?? "",
      vehicleModel: vehicle?.model ?? "",
      vehicleYear: vehicle?.year != null ? String(vehicle.year) : "",
      vehiclePlate: vehicle?.plate ?? "",
      vehicleColor: vehicle?.color ?? "",
      vehicleFuelType: vehicle?.fuelType ?? "",
      vehicleChassis: vehicle?.chassisNumber ?? "",
      vehicleEngine: vehicle?.engineNumber ?? "",
      roadTaxExpiry: toISODate(vehicle?.roadTaxExpiry ?? null),
      insuranceExpiry: toISODate(vehicle?.insuranceExpiry ?? null),
      lastServiceDate: toISODate(vehicle?.lastServiceDate ?? null),
      nextServiceDate: toISODate(vehicle?.nextServiceDate ?? null),
      currentMileage: vehicle?.currentMileage != null ? String(vehicle.currentMileage) : "",
    };

    return res.json({ ok: true, fields });
  } catch (err) {
    return next(err);
  }
}