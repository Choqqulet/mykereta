import app from "./app.js";

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`API listening on 0.0.0.0:${port}`);
});