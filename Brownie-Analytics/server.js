const express = require('express');
const port = process.env.PORT || 1611
const srv = express()

srv.use(express.json())
srv.use(express.urlencoded({extended: true}))

srv.use('/', express.static(__dirname + "/home"))

srv.use('/analytics', express.static(__dirname + "/analytics"))

srv.listen(port)

