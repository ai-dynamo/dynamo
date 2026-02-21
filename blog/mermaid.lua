local counter = 0
local img_dir = "mermaid_images"

local function ensure_dir()
    os.execute("mkdir -p " .. img_dir)
end

function CodeBlock(block)
    if block.classes[1] ~= "mermaid" then
        return nil
    end

    ensure_dir()
    counter = counter + 1

    local infile = img_dir .. "/diagram_" .. counter .. ".mmd"
    local outfile = img_dir .. "/diagram_" .. counter .. ".png"

    local f = io.open(infile, "w")
    f:write(block.text)
    f:close()

    local cmd = string.format(
        "mmdc -i %s -o %s -b white -w 1200 -s 2 2>&1",
        infile, outfile
    )
    local handle = io.popen(cmd)
    local result = handle:read("*a")
    handle:close()

    local img = io.open(outfile, "rb")
    if not img then
        io.stderr:write("mermaid render failed: " .. result .. "\n")
        return nil
    end
    img:close()

    return pandoc.Para({
        pandoc.Image({}, outfile, ""),
    })
end
