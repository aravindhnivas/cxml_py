import { spawn } from 'child_process'
import path from 'path'
import { $ } from "bun";

try {
    await $`rm -rf build`
    await $`rm -rf dist`
    await $`rm umdapy.spec`    
} catch (error) {
    console.log('No build or dist directory')
}


const start_time = new Date()

const maindir = path.resolve("../src")
const mainfile = path.join(maindir, 'main.py')

const opts = '--noconfirm --onedir --console --debug noarchive --noupx'
const name = `--name umdapy`

const icon = `--icon ${path.join(maindir, 'icons/icon.ico')}`
const hooks = `--additional-hooks-dir ${path.join(maindir, 'hooks')}`
const hiddenimport = '--hidden-import umdalib'

const args =
    `${opts} ${name} ${icon} ${hooks} ${hiddenimport} ${mainfile}`.split(
        ' '
    ).filter(f => f.trim() !== '')

console.log(args)

const py = spawn("pyinstaller", args)
py.stdout.on('data', (data) => console.log(data.toString('utf8')))
py.stderr.on('data', (data) => console.log(data.toString('utf8')))
py.on('close', async () => {
    console.log('Pyinstaller done in', new Date() - start_time, 'ms')
    // await $`cd dist && zip -r9 umdapy-darwin.zip umdapy/`
})
py.on('error', (err) => console.log('error occured', err))
