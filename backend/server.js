import app from './src/app.js';
import logger from './src/utils/logger.js';

const port = process.env.PORT || 3000;

app.listen(port, () => {
  logger.info(`API listening on 0.0.0.0:${port}`);
});
