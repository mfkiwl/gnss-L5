% write_l5i_codes.m
%
% Generates GPS L5 I5 codes for PRN 1-32 and writes them to a binary file
% as raw uint8 (0/1) chips, 10230 bytes per PRN, PRNs in order.
%
% based on CU-GNSS-SDR's generateL5Icode function, which is based on the GPS Interface Control
% Run from the GPS_L5C directory:
%   octave write_l5i_codes.m

addpath include
addpath Common

% Minimal settings struct — only codeLength is needed by generateL5Icode
settings.codeLength = 10230;

OUTPUT_FILE = 'l5i_codes_octave.bin';
NUM_PRNS    = 32;

fid = fopen(OUTPUT_FILE, 'wb');
if fid < 0
    error('Could not open output file: %s', OUTPUT_FILE);
end

for prn = 1:NUM_PRNS
    % generateL5Icode returns a ±1 sequence (+1 = binary 0, -1 = binary 1)
    code_bipolar = generateL5Icode(prn, settings);

    % Convert ±1 to binary 0/1 to match Python's chips array convention
    code_binary = uint8((1 - code_bipolar) / 2);

    fwrite(fid, code_binary, 'uint8');
end

fclose(fid);
fprintf('Wrote %d codes (%d chips each) to %s\n', NUM_PRNS, settings.codeLength, OUTPUT_FILE);
