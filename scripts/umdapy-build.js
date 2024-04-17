import { spawn } from 'child_process'
import path from 'path'
// import fs from 'fs-extra'
const maindir = path.resolve("../src")
const icon = path.join(maindir, 'icons/icon.ico')
const hooks = path.join(maindir, 'hooks')
const mainfile = path.join(maindir, 'main.py')
const args =
    `--noconfirm --onedir --console --icon ${icon} --name umdapy --debug noarchive --noupx --additional-hooks-dir ${hooks} --hidden-import umdalib --paths ${maindir} ${mainfile}`.split(
        ' '
    )
console.log(args)

const py = spawn("pyinstaller", args)
py.stdout.on('data', (data) => console.log(data.toString('utf8')))
py.stderr.on('data', (data) => console.log(data.toString('utf8')))
py.on('close', () => {
    console.log('pyinstaller done')
})
py.on('error', (err) => console.log('error occured', err))