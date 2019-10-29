"""
Redshift catalogues for different data types in C-EAGLE.
"""
def group_data():
    redshift_catalogue = {
        'z_value': # The sequential number of the sna/ipshots
             ['z014p003', 'z006p772', 'z004p614', 'z003p512', 'z002p825',
             'z002p348', 'z001p993', 'z001p716', 'z001p493', 'z001p308',
             'z001p151', 'z001p017', 'z000p899', 'z000p795', 'z000p703',
             'z000p619', 'z000p543', 'z000p474', 'z000p411', 'z000p366',
             'z000p352', 'z000p297', 'z000p247', 'z000p199', 'z000p155',
             'z000p113', 'z000p101', 'z000p073', 'z000p036', 'z000p000'],
        'z_IDNumber': # The value of the sna/ipshot redshifts
             ['000', '001', '002', '003', '004', '005', '006',
             '007', '008', '009', '010', '011', '012', '013',
             '014', '015', '016', '017', '018', '019', '020',
             '021', '022', '023', '024', '025', '026', '027',
             '028', '029']
         }
    return redshift_catalogue

def particle_data():
    """
    Redshift catalogues for group data and particle data
    are the same.
    """
    return group_data()

def snipshots_data():
    z_dict = {
        'z_type': # either 'snapshot' or 'snipshot'
            ['snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot',
            'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot',
            'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot',
            'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot',
            'snapshot', 'snapshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot',
            'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot'],
        'z_IDNumber': # The sequential number of the sna/ipshots
            ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011',
            '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023',
            '024', '025', '026', '027', '028', '029', '000', '001', '002', '003', '004', '005',
            '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017',
            '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029',
            '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041',
            '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053',
            '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065',
            '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077',
            '078', '079', '080', '081'],
        'z_value': # The value of the sna/ipshot redshifts
            ['z014p003', 'z006p772', 'z004p614', 'z003p512', 'z002p825', 'z002p348', 'z001p993',
            'z001p716', 'z001p493', 'z001p308', 'z001p151', 'z001p017', 'z000p899', 'z000p795',
            'z000p703', 'z000p619', 'z000p543', 'z000p474', 'z000p411', 'z000p366', 'z000p352',
            'z000p297', 'z000p247', 'z000p199', 'z000p155', 'z000p113', 'z000p101', 'z000p073',
            'z000p036', 'z000p000', 'z010p873', 'z008p988', 'z007p708', 'z006p052', 'z005p478',
            'z005p008', 'z004p279', 'z003p989', 'z003p736', 'z003p313', 'z003p134', 'z002p972',
            'z002p691', 'z002p567', 'z002p453', 'z002p250', 'z002p158', 'z002p073', 'z001p917',
            'z001p846', 'z001p779', 'z001p656', 'z001p599', 'z001p544', 'z001p443', 'z001p396',
            'z001p351', 'z001p266', 'z001p226', 'z001p188', 'z001p116', 'z001p082', 'z001p049',
            'z000p986', 'z000p956', 'z000p927', 'z000p872', 'z000p846', 'z000p820', 'z000p771',
            'z000p748', 'z000p725', 'z000p681', 'z000p660', 'z000p639', 'z000p599', 'z000p580',
            'z000p562', 'z000p525', 'z000p508', 'z000p491', 'z000p458', 'z000p442', 'z000p426',
            'z000p395', 'z000p381', 'z000p366', 'z000p338', 'z000p324', 'z000p311', 'z000p284',
            'z000p272', 'z000p259', 'z000p234', 'z000p223', 'z000p211', 'z000p188', 'z000p177',
            'z000p166', 'z000p144', 'z000p133', 'z000p123', 'z000p103', 'z000p093', 'z000p083',
            'z000p063', 'z000p054', 'z000p045', 'z000p026', 'z000p018', 'z000p009', 'z000p000']
    }
    return z_dict
