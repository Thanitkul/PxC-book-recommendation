const http = require("http");

http.createServer((req, res) => {
  const host = req.headers.host.replace(/:\d+$/, ":8443"); // replace port if needed
  res.writeHead(301, { Location: `https://${host}${req.url}` });
  res.end();
}).listen(8080, () => {
  console.log("🌐 HTTP Redirector running on http://localhost:8080 → https://localhost:8443");
});
