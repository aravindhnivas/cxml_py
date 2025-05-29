import { spawn } from 'child_process'
import path from 'path'
import { $ } from "bun";
import fs from 'fs'

try {
    await $`rm -rf build`
    await $`rm -rf dist`
    await $`rm cxml_py.spec`    
} catch (error) {
    console.log('No build or dist directory', error)
}


const start_time = new Date()

const maindir = path.resolve("../src")
const mainfile = path.join(maindir, 'main.py')

const opts = '--noconfirm --onedir --console --debug noarchive --noupx'
const name = `--name cxml_py`

const icon = `--icon ${path.join(maindir, 'icons/icon.ico')}`
const hooks = `--additional-hooks-dir ${path.join(maindir, 'hooks')}`
const hiddenimport = '--hidden-import cxml_lib'

const args =
    `${opts} ${name} ${icon} ${hooks} ${hiddenimport} ${mainfile}`.split(
        ' '
    ).filter(f => f.trim() !== '')

console.log(args)

const py = spawn("pyinstaller", args)

// save stdout and stderr to files
const stdout_file = path.join('stdout.txt')
const stderr_file = path.join('stderr.txt')

py.stdout.pipe(fs.createWriteStream(stdout_file))
py.stderr.pipe(fs.createWriteStream(stderr_file))

// py.stdout.on('data', (data) => console.log(data.toString('utf8')))
// py.stderr.on('data', (data) => console.log(data.toString('utf8')))

py.on('close', async () => {
    const time_in_ms = new Date() - start_time
    const time_in_minutes = time_in_ms / 60000
    console.log(`Pyinstaller done in ${time_in_minutes} minutes`)
    // await $`cd dist && zip -r9 cxml_py-darwin.zip cxml_py/`
})
py.on('error', (err) => console.log('error occured', err))
